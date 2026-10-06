"""Preserve receptor chain identities after binder prep, which merges target IDs."""

import numpy as np


def install_receptor_chains(model, chains):
    chains = chains.split(",") if isinstance(chains, str) else list(chains)
    if len(chains) == 1:
        return
    idx = np.asarray(model._pdb["idx"]["chain"][: model._target_len])
    lengths = [int((idx == c).sum()) for c in chains]
    assert all(lengths)
    assert sum(lengths) == model._target_len
    model._lengths = lengths + [model._binder_len]
    asym = np.concatenate([np.full(n, i) for (i, n) in enumerate(model._lengths)])
    aa = np.asarray(model._pdb["batch"]["aatype"][: model._target_len])
    entities = []
    seqs = []
    start = 0
    syms = []
    for n in lengths:
        s = tuple(aa[start : start + n])
        start += n
        if s not in seqs:
            seqs.append(s)
        ent = seqs.index(s)
        syms.append(entities.count(ent))
        entities.append(ent)
    entities.append(len(seqs))
    syms.append(0)
    model._inputs.update(
        asym_id=asym,
        entity_id=np.repeat(entities, model._lengths),
        sym_id=np.repeat(syms, model._lengths),
    )
    model._score_target_chains = tuple(chains)
