"""Public PDB/JSON input contract and coordinate-preserving outputs."""

import copy
import json
from pathlib import Path
import torch
from Bio.PDB import PDBParser

from dcflow.collate import PaddingCollate
from dcflow.geometry.protein.parsers import parse_pdb
from dcflow.geometry.protein.constants import resindex_to_ressymb
from dcflow.geometry.protein.writers import save_pdb
from dcflow.model.cyclic import attach_cyclic_features
from dcflow.model.torsion import (
    full_atom_reconstruction,
    get_torsion_angle,
    get_heavyatom_mask,
)


def read_case(path):
    path = Path(path).resolve()
    case = json.loads(path.read_text())
    for key in ["pocket", "peptide_reference", "receptor", "receptor_msa"]:
        if key in case:
            case[key] = str((path.parent / case[key]).resolve())
    case.setdefault("topology", "HT")
    case.setdefault("receptor_chains", "A")
    if case["topology"] not in ("HT", "SS"):
        raise ValueError("topology must be HT or SS")
    return case


def prepare(case):
    rec = parse_pdb(case["pocket"])[0]
    if rec is None:
        raise ValueError("Pocket contains no complete backbone residues")
    if "peptide_reference" in case:
        pep = parse_pdb(case["peptide_reference"])[0]
        if pep is None:
            raise ValueError("Invalid peptide reference")
        length = len(pep["aa"])
        center = pep["pos_heavyatom"][:, 1].mean(0)
    else:
        length = int(case["length"])
        center = torch.tensor(case["center"], dtype=torch.float32)
        # Neutral feature scaffold: coordinates and sequence are masked by the model.
        aa = torch.zeros(length, dtype=torch.long)
        r = torch.eye(3).expand(1, length, 3, 3)
        x = torch.zeros(1, length, 3)
        xyz, _, _ = full_atom_reconstruction(r, x, torch.zeros(1, length, 5), aa[None])
        pep = dict(
            aa=aa,
            pos_heavyatom=torch.nn.functional.pad(xyz[0], (0, 0, 0, 1)) + center,
            mask_heavyatom=get_heavyatom_mask(aa[None])[0],
            chain_id=["P"] * length,
            chain_nb=torch.zeros(length, dtype=torch.long),
            resseq=torch.arange(1, length + 1),
            res_nb=torch.arange(1, length + 1),
            icode=[" "] * length,
        )
    if length < 3 or center.shape != (3,) or not torch.isfinite(center).all():
        raise ValueError("Invalid peptide length or center")
    case["length"] = length
    for item in (rec, pep):
        item["pos_heavyatom"] -= center[None, None]
        item["torsion_angle"], item["torsion_angle_mask"] = get_torsion_angle(
            item["pos_heavyatom"], item["aa"]
        )
    rec["chain_nb"] += int(pep["chain_nb"].max()) + 1
    data = dict(
        generate_mask=torch.cat(
            [torch.zeros(len(rec["aa"])), torch.ones(length)]
        ).bool()
    )
    for key in rec:
        data[key] = (
            torch.cat([rec[key], pep[key]])
            if torch.is_tensor(rec[key])
            else rec[key] + pep[key]
        )
    data = attach_cyclic_features(
        data, {"HT": "HeadToTail", "SS": "CysToCys"}[case["topology"]], terminal_ss=True
    )
    return data, center


def make_batch(item, count, device):
    result = PaddingCollate(eight=False)([copy.deepcopy(item) for _ in range(count)])
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in result.items()}


def write_outputs(final, item, center, case, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    final = {k: v.detach().cpu() for k, v in final.items()}
    xyz, _, _ = full_atom_reconstruction(
        final["rotmats"], final["trans"], final["angles"], final["seqs"]
    )
    if not torch.isfinite(xyz).all():
        raise ValueError("Nonfinite generated coordinates")
    mask = item["generate_mask"]
    chains = case["receptor_chains"].split(",")
    peptide_chain = next(c for c in "PQRSTUVWXYZABCDEFGHIJKLMNO" if c not in chains)
    receptor_lines = [
        s
        for s in Path(case.get("receptor", case["pocket"])).read_text().splitlines()
        if s.startswith("ATOM  ") and s[21] in chains
    ]
    if not receptor_lines:
        raise ValueError("No requested receptor chains in output receptor")
    records = []
    for i in range(len(xyz)):
        seq = final["seqs"][i, mask]
        sequence = "".join(resindex_to_ressymb[int(a)] for a in seq)
        candidate = f"candidate_{i:03d}"
        peptide = output / f"{candidate}_peptide.pdb"
        n = len(seq)
        save_pdb(
            dict(
                chain_nb=torch.zeros(n).long(),
                chain_id=[peptide_chain] * n,
                resseq=torch.arange(1, n + 1),
                icode=[" "] * n,
                aa=seq,
                pos_heavyatom=torch.nn.functional.pad(
                    xyz[i, mask] + center, (0, 0, 0, 1)
                ),
                mask_heavyatom=get_heavyatom_mask(seq[None])[0],
            ),
            path=str(peptide),
        )
        peptide_lines = [
            s for s in peptide.read_text().splitlines() if s.startswith("ATOM  ")
        ]
        lines = []
        for serial, line in enumerate(receptor_lines + peptide_lines, 1):
            lines.append(line[:6] + f"{serial:5d}" + line[11:])
        complex_path = output / f"{candidate}.pdb"
        complex_path.write_text("\n".join(lines + ["TER", "END"]) + "\n")
        record = dict(
            candidate=candidate,
            peptide_sequence=sequence,
            structure=complex_path.name,
            peptide_structure=peptide.name,
            peptide_chain=peptide_chain,
            topology=case["topology"],
        )
        records.append(record)
    (output / "candidates.json").write_text(json.dumps(records, indent=2) + "\n")
    return records
