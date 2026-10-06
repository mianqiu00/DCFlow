#!/usr/bin/env python3
"""DCFlow generation, guided refinement and cascaded search."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

from dcflow.runtime import load_model, seed_all
from dcflow.inputs import read_case, prepare, make_batch, write_outputs
from dcflow.sampling import Sampler, schedule
from dcflow.steering import select, quality_select
from dcflow.guidance import AFCycGuidance
from dcflow.adapters.client import AFCycClient


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--mode", choices=["direct", "guided", "refine", "cascade"], default="direct"
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int)
    p.add_argument("--num-designs", type=int)
    p.add_argument("--num-steps", type=int)
    p.add_argument(
        "--settings", type=Path, default=Path(__file__).parent / "config/inference.json"
    )
    p.add_argument("--mpnn-source")
    p.add_argument("--mpnn-weights")
    p.add_argument("--folding-python", default=os.environ.get("FOLDING_PYTHON"))
    p.add_argument("--af-params", default=os.environ.get("ALPHAFOLD_PARAMS"))
    p.add_argument("--colabdesign-source", default=os.environ.get("COLABDESIGN_SOURCE"))
    p.add_argument("--oracle-gpu", help="CUDA_VISIBLE_DEVICES for the external worker")
    p.add_argument(
        "--no-guidance",
        action="store_true",
        help="Disable trajectory feedback while retaining requested refinement/search",
    )
    args = p.parse_args()
    settings = json.loads(args.settings.read_text())
    for key in ["seed", "num_designs", "num_steps"]:
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    stages = settings["refinement"]["iterations"]
    if len(stages) != 3 or any(not isinstance(n, int) or n < 1 for n in stages):
        p.error("Refinement requires three positive integer stage lengths")
    if not 1 <= settings["refinement"]["warmup"] <= stages[0]:
        p.error("Screening warmup must fit within the first refinement stage")
    if settings["refinement"]["patience"] < 1:
        p.error("Early-stopping patience must be positive")
    if settings["oracle"]["offset_type"] != 2:
        p.error("This inference protocol uses cyclic offset type 2")
    if settings["num_designs"] < 1 or settings["num_steps"] < 4:
        p.error("Use a positive candidate count and at least four time points")
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Use a new or empty output directory")
    case = read_case(args.input)
    if args.mode != "direct":
        if case["topology"] != "HT":
            p.error(
                "Cyclic cofolding modes require HT topology; direct generation supports HT and SS"
            )
        if not args.folding_python or not args.af_params:
            p.error("Set FOLDING_PYTHON and ALPHAFOLD_PARAMS for cofolding modes")
        if (
            not args.no_guidance
            and settings["guidance"]["steps"]
            and max(settings["guidance"]["steps"]) >= settings["num_steps"]
        ):
            p.error("Guide event steps must lie within the integration loop")
    search = settings["search"]
    if args.mode == "cascade" and not (
        1 <= search["K0"] <= search["K"]
        and 1 <= search["N0"] <= search["K0"] * search["M"]
    ):
        p.error("Invalid cascade budget")
    args.output.mkdir(parents=True, exist_ok=True)
    item, center = prepare(case)
    # Runtime files intentionally resolve user inputs; distributed examples are relative.
    (args.output / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    command = [
        args.folding_python or sys.executable,
        "-m",
        "dcflow.adapters.afcyc_worker",
        "--case",
        str((args.output / "case.json").resolve()),
        "--settings",
        str((args.output / "settings.json").resolve()),
        "--params",
        args.af_params or "",
    ]
    if args.colabdesign_source:
        command += ["--colabdesign-source", args.colabdesign_source]
    env = {
        "PYTHONPATH": str(Path(__file__).resolve().parent)
        + os.pathsep
        + os.environ.get("PYTHONPATH", "")
    }
    if args.oracle_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.oracle_gpu
    seed_all(settings["seed"])
    model = load_model(
        device=args.device, mpnn_source=args.mpnn_source, mpnn_weights=args.mpnn_weights
    )
    context = nullcontext(None) if args.mode == "direct" else AFCycClient(command, env)
    with context as oracle:
        cascade = args.mode == "cascade"
        count = search["K"] if cascade else settings["num_designs"]
        sampler = Sampler(model, make_batch(item, count, args.device))
        state = sampler.initialize()
        times = schedule(
            settings["num_steps"], model.sequence_start_t if cascade else None
        )
        roots = list(range(count))
        branch_indices = roots
        offset = 0
        if cascade:
            boundary = int(np.argmin(abs(times - model.sequence_start_t)))
            state = sampler.run(state, times[: boundary + 1])
            roots = select(
                state.translations[:, item["generate_mask"]].cpu().numpy(),
                search["K0"],
                settings["seed"] + 1,
                search["sigma"],
                search["temperature"],
                search["epsilon"],
            )
            branch_indices = [i for i in roots for _ in range(search["M"])]
            child = Sampler(model, make_batch(item, len(branch_indices), args.device))
            state = child.branch(state, branch_indices)
            sampler = child
            times = times[boundary:]
            offset = boundary
        guide = None
        if oracle is not None and not args.no_guidance:
            g = settings["guidance"]
            guide = AFCycGuidance(
                oracle,
                sampler.mask[0],
                g["steps"],
                g["scale"],
                g["positions"],
                g["alternatives"],
            )
        state = sampler.run(state, times, guide, offset)
        final = sampler.finish(state, guide)
        records = write_outputs(final, item, center, case, args.output / "generated")
        summary = dict(
            mode=args.mode,
            seed=settings["seed"],
            selected_roots=roots,
            branch_roots=branch_indices,
            generated_count=len(records),
            guidance_events=[] if guide is None else guide.events,
            guidance_enabled=guide is not None,
            settings=settings,
        )
        sequences = [r["peptide_sequence"] for r in records]
        if args.mode in ("refine", "cascade"):
            tokens = None
            if cascade:
                screen = oracle.request(
                    "screen", sequences=sequences, seed=settings["seed"] + 1000
                )["results"]
                retained = quality_select(
                    [r["Q"] for r in screen],
                    search["N0"],
                    settings["seed"] + 2,
                    search["quality_temperature"],
                )
                tokens = [screen[i]["token"] for i in retained]
                sequences = [sequences[i] for i in retained]
                summary.update(screen=screen, retained_candidates=retained)
            refined = oracle.request(
                "refine",
                sequences=sequences,
                seed=settings["seed"] + 1000,
                output=str((args.output / "refined").resolve()),
                tokens=tokens,
                early_stop=cascade,
            )["results"]
            summary["refined_count"] = len(refined)
            summary["refinement_updates"] = [r["updates"] for r in refined]
            summary["final_sequences"] = [r["peptide_sequence"] for r in refined]
        else:
            summary["final_sequences"] = sequences
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                output=str(args.output),
                generated=len(records),
                final=len(summary["final_sequences"]),
            )
        )
    )


if __name__ == "__main__":
    main()
