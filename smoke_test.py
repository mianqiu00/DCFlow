"""Real-weight integration smoke; --with-afcyc includes the external oracle."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--with-afcyc", action="store_true")
    p.add_argument("--oracle-gpu")
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    settings = json.loads((root / "config/inference.json").read_text())
    settings.update(num_steps=12, num_designs=2)
    settings["guidance"].update(steps=[8], positions=1, alternatives=1)
    settings["search"].update(K=3, K0=1, M=2, N0=1)
    settings["refinement"].update(iterations=[5, 3, 1], warmup=5, patience=1)
    path = args.output / "smoke_settings.json"
    path.write_text(json.dumps(settings, indent=2))
    base = [sys.executable, str(root / "inference.py")]
    commands = []

    def run(case, name, mode):
        command = base + [
            str(case),
            "--output",
            str(args.output / name),
            "--settings",
            str(path),
            "--device",
            args.device,
            "--mode",
            mode,
        ]
        if args.oracle_gpu is not None:
            command += ["--oracle-gpu", args.oracle_gpu]
        subprocess.run(command, check=True, cwd=root)
        commands.append(dict(name=name, mode=mode))

    run(root / "examples/5LSO.json", "ht", "direct")
    case = json.loads((root / "examples/5LSO.json").read_text())
    for key in ["pocket", "peptide_reference", "receptor", "receptor_msa"]:
        case[key] = str((root / "examples" / case[key]).resolve())
    case["topology"] = "SS"
    ss = args.output / "ss_input.json"
    ss.write_text(json.dumps(case))
    run(ss, "ss", "direct")
    records = json.loads((args.output / "ss/generated/candidates.json").read_text())
    assert all(
        r["peptide_sequence"][0] == r["peptide_sequence"][-1] == "C" for r in records
    )
    if args.with_afcyc:
        run(root / "examples/5LSO.json", "cascade", "cascade")
        summary = json.loads((args.output / "cascade/summary.json").read_text())
        assert summary["generated_count"] == 2 and summary["refined_count"] == 1
        assert len(summary["guidance_events"]) == 1 and len(summary["screen"]) == 2
        assert summary["refinement_updates"][0] >= 6
    report = dict(
        passed=True,
        checks=commands,
        real_dcflow_weights=True,
        real_mpnn=True,
        real_afcyc=args.with_afcyc,
        seconds=time.time() - started,
    )
    (args.output / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
