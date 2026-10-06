"""Independent fixed-sequence, receptor-only five-model cofolding."""

import argparse
import json
import os
from pathlib import Path
from dcflow.inputs import read_case, prepare
from dcflow.adapters.client import AFCycClient


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--folding-python", default=os.environ.get("FOLDING_PYTHON"))
    p.add_argument("--af-params", default=os.environ.get("ALPHAFOLD_PARAMS"))
    p.add_argument("--colabdesign-source", default=os.environ.get("COLABDESIGN_SOURCE"))
    args = p.parse_args()
    if not args.folding_python or not args.af_params:
        p.error("Configure external folding Python and AF parameters")
    case = read_case(args.input)
    prepare(case)
    if case["topology"] != "HT":
        p.error("This cyclic cofolding protocol uses HT topology")
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "case.json").write_text(json.dumps(case))
    root = Path(__file__).parent
    settings = json.loads((root / "config/inference.json").read_text())
    sequences = json.loads((args.results / "summary.json").read_text())[
        "final_sequences"
    ]
    command = [
        args.folding_python,
        "-m",
        "dcflow.adapters.afcyc_worker",
        "--case",
        str((args.output / "case.json").resolve()),
        "--params",
        args.af_params,
        "--settings",
        str((root / "config/inference.json").resolve()),
        "--models",
        "1,2,3,4,5",
    ]
    if args.colabdesign_source:
        command += ["--colabdesign-source", args.colabdesign_source]
    results = []
    with AFCycClient(
        command,
        {
            "PYTHONPATH": str(root.resolve())
            + os.pathsep
            + os.environ.get("PYTHONPATH", "")
        },
    ) as worker:
        for index in range(5):
            results.append(
                worker.request(
                    "evaluate",
                    sequences=sequences,
                    recycles=3,
                    model_index=index,
                    output=str(args.output.resolve()),
                )["results"]
            )
    aggregate = []
    for i, seq in enumerate(sequences):
        avg = {k: sum(result[i][k] for result in results) / 5 for k in results[0][i]}
        avg["interface_composite"] = (
            avg["ipTM"] + avg["peptide_mean_pLDDT"] / 100 + 1 - avg["iPAE_normalized"]
        )
        avg["strict_hit"] = (
            avg["ipTM"] >= 0.7
            and avg["peptide_mean_pLDDT"] >= 70
            and avg["iPAE_normalized"] <= 0.3
        )
        aggregate.append(dict(sequence=seq, **avg))
    (args.output / "metrics.json").write_text(
        json.dumps(dict(per_model=results, mean=aggregate), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
