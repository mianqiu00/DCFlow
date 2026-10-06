"""External ColabDesign adapter: cyclic scoring and resumable refinement."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path
import numpy as np


def cyclic_offset(length):
    indices = np.arange(length)
    linear = indices[:, None] - indices[None, :]
    wrapped = np.stack([indices, indices + length], axis=-1)
    cyclic = abs(wrapped[:, None, :, None] - wrapped[None, :, None, :]).min(axis=(2, 3))
    cyclic[cyclic < abs(linear)] *= -1
    return cyclic * np.sign(linear)


def metrics(model, hotspots):
    aux = model.aux
    pae = np.asarray(aux["pae"])
    n = model._target_len
    binder = np.arange(n, n + model._binder_len)
    target = np.arange(n)
    cross = lambda ids: float(
        (pae[np.ix_(binder, ids)].mean() + pae[np.ix_(ids, binder)].mean()) / 62.0
    )
    out = dict(
        ipTM=float(aux["i_ptm"]),
        peptide_mean_pLDDT=float(np.asarray(aux["plddt"])[binder].mean() * 100),
        iPAE_normalized=cross(target),
        hotspot_iPAE_normalized=cross(hotspots),
    )
    out["composite"] = (
        out["ipTM"]
        + out["peptide_mean_pLDDT"] / 100
        + 1
        - out["hotspot_iPAE_normalized"]
    )
    out["Q"] = 0.75 * (1 - out["hotspot_iPAE_normalized"]) + 0.25 * out["ipTM"]
    return out


class Backend:
    def __init__(self, case, params, settings, model_names=None):
        from colabdesign import mk_afdesign_model
        import jax.numpy as jnp
        from dcflow.adapters.receptor_msa import ReceptorMSA
        from dcflow.adapters.multichain_binder import install_receptor_chains

        self.settings = settings
        self.cache = {}
        self.states = {}
        self.hotspots = np.array([], dtype=int)
        msa = ReceptorMSA(settings["oracle"]["msa_rows"])
        msa.load(Path(case["receptor_msa"]), case["length"])

        def objectives(aux):
            n = self.model._target_len
            pae = aux["pae"]
            h = jnp.asarray(self.hotspots)
            hotspot = (pae[n:, h].mean() + pae[h, n:].mean()) / 62.0
            return {"dc_iptm": 1 - aux["i_ptm"], "dc_hotspot_pae": hotspot}

        self.model = mk_afdesign_model(
            protocol="binder",
            use_multimer=True,
            data_dir=params,
            model_names=model_names or [settings["oracle"]["model"]],
            pre_callback=msa.callback,
            loss_callback=objectives,
        )
        m = self.model
        m.prep_inputs(
            pdb_filename=case["receptor"],
            target_chain=case["receptor_chains"],
            binder_len=case["length"],
            hotspot=case["hotspots"],
            use_binder_template=False,
            use_initial_guess=False,
        )
        msa.install(m)
        install_receptor_chains(m, case["receptor_chains"])
        requested = set()
        for value in case["hotspots"].split(","):
            value = value.strip()
            if value:
                requested.add((value[0], int(value[1:].lstrip(":"))))
        idx = m._pdb["idx"]
        self.hotspots = np.array(
            [
                i
                for i, (c, r) in enumerate(
                    zip(idx["chain"][: m._target_len], idx["residue"][: m._target_len])
                )
                if (str(c), int(r)) in requested
            ],
            dtype=int,
        )
        if not requested or len(self.hotspots) != len(requested):
            raise ValueError("All hotspot selectors must resolve on the receptor")
        residue = np.asarray(m._inputs["residue_index"])
        offset = residue[:, None] - residue[None, :]
        offset[m._target_len :, m._target_len :] = cyclic_offset(m._binder_len)
        m._inputs["offset"] = offset
        # Register callback terms before the first JIT and before restart snapshots.
        m.opt["weights"].update(dc_iptm=1.0, dc_hotspot_pae=1.0)
        m._opt["weights"].update(dc_iptm=1.0, dc_hotspot_pae=1.0)

    def evaluate(self, sequences, recycles=None, model_index=0, output=None):
        results = []
        for i, seq in enumerate(sequences):
            if len(seq) != self.model._binder_len or set(seq) - set(
                "ACDEFGHIKLMNPQRSTVWY"
            ):
                raise ValueError("Invalid binder sequence")
            key = (seq, recycles, model_index)
            if key not in self.cache or output:
                self.model.predict(
                    seq=seq,
                    models=[model_index],
                    num_recycles=(
                        recycles
                        if recycles is not None
                        else self.settings["oracle"]["num_recycles"]
                    ),
                    sample_models=False,
                    dropout=False,
                    verbose=False,
                    seed=0,
                )
                self.cache[key] = metrics(self.model, self.hotspots)
                if output:
                    self.model.save_pdb(
                        str(
                            Path(output)
                            / f"candidate_{i:03d}_model_{model_index+1}.pdb"
                        )
                    )
            results.append(self.cache[key])
        return results

    def weights(self, screen=False):
        m = self.model
        weights = {k: 0.0 for k in m.opt["weights"]}
        weights.update(
            dc_iptm=self.settings["refinement"]["iptm_weight"],
            dc_hotspot_pae=(
                0.0 if screen else self.settings["refinement"]["hotspot_pae_weight"]
            ),
        )
        m.opt["weights"].update(weights)

    def current_sequence(self):
        """Decode the updated optimizer parameters, including the final update."""
        from colabdesign.shared.model import soft_seq
        from colabdesign.af.alphafold.common import residue_constants

        m = self.model
        sequence = soft_seq(m._params["seq"], m._inputs["bias"], m.opt)
        indices = np.asarray(sequence["hard"])[0].argmax(-1)
        return "".join(residue_constants.restypes[int(i)] for i in indices)

    def step(self, index, screen=False):
        m = self.model
        first, soft, hard = self.settings["refinement"]["iterations"]
        if index < first:
            stage = "logits"
            s = (index + 1) / first
            t = 1.0
            h = 0.0
            dropout = True
        elif index < first + soft:
            stage = "soft"
            s = 1.0
            t = 0.01 + 0.99 * (1 - (index - first + 1) / soft) ** 2
            h = 0.0
            dropout = True
        else:
            stage = "hard"
            s = 1.0
            t = 0.01
            h = 1.0
            dropout = False
        self.weights(screen)
        m.set_opt(soft=s, temp=t, hard=h, dropout=dropout)
        m.step(
            lr_scale=(1 - s) + s * t,
            num_recycles=self.settings["oracle"]["num_recycles"],
            models=[0],
            sample_models=False,
            save_best=False,
            verbose=0,
        )
        result = metrics(m, self.hotspots)
        result.update(
            step=index + 1, stage=stage, sequence=m.get_seq(get_best=False)[0]
        )
        # Common composite used explicitly for the soft-stage stopping policy.
        m.aux["log"]["composite"] = result["composite"]
        return result

    def capture(self):
        import jax

        # Snapshot all mutable design state, including optimizer moments and RNG key.
        keys = (
            "_params",
            "_state",
            "_inputs",
            "opt",
            "_tmp",
            "_k",
            "_key",
            "key",
            "aux",
        )
        state = {
            k: copy.deepcopy(jax.device_get(getattr(self.model, k)))
            for k in keys
            if hasattr(self.model, k)
        }
        state["_numpy_rng"] = np.random.get_state()
        return state

    def restore(self, state):
        for key, value in state.items():
            if key == "_numpy_rng":
                np.random.set_state(value)
            else:
                setattr(self.model, key, copy.deepcopy(value))

    def screen(self, sequences, seed):
        self.states = {}
        results = []
        for i, sequence in enumerate(sequences):
            self.model.restart(seed=seed + i, seq=sequence)
            trace = [
                self.step(j, screen=True)
                for j in range(self.settings["refinement"]["warmup"])
            ]
            self.states[str(i)] = (self.capture(), trace)
            results.append(dict(token=str(i), **trace[-1]))
        return results

    def refine(self, sequences, seed, output, tokens=None, early_stop=False):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        first, soft, hard = self.settings["refinement"]["iterations"]
        results = []
        for i, sequence in enumerate(sequences):
            if tokens is not None:
                state, trace = self.states.pop(str(tokens[i]))
                self.restore(state)
                trace = copy.deepcopy(trace)
                start = self.settings["refinement"]["warmup"]
            else:
                self.model.restart(seed=seed + i, seq=sequence)
                trace = []
                start = 0
            best = -np.inf
            stale = 0
            stopped = False
            index = start
            while index < first + soft + hard:
                entry = self.step(index)
                trace.append(entry)
                if early_stop and entry["stage"] == "soft":
                    if (
                        entry["composite"]
                        > best + self.settings["refinement"]["min_delta"]
                    ):
                        best = entry["composite"]
                        stale = 0
                    else:
                        stale += 1
                    if stale >= self.settings["refinement"]["patience"]:
                        stopped = True
                        index = first + soft
                        continue
                index += 1
            seq = self.current_sequence()
            # Return a fixed discrete sequence prediction, not a soft-state structure.
            result = self.evaluate([seq])[0]
            self.model.predict(
                seq=seq,
                models=[0],
                num_recycles=self.settings["oracle"]["num_recycles"],
                sample_models=False,
                dropout=False,
                verbose=False,
                seed=0,
            )
            self.model.save_pdb(str(output / f"candidate_{i:03d}.pdb"))
            result = dict(
                candidate=f"candidate_{i:03d}",
                peptide_sequence=seq,
                early_stopped=stopped,
                updates=len(trace),
                metrics=result,
                trajectory=trace,
            )
            (output / f"candidate_{i:03d}.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            results.append(result)
        self.states.clear()
        return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--params", required=True)
    p.add_argument("--settings", required=True)
    p.add_argument("--colabdesign-source")
    p.add_argument("--models", default=None)
    args = p.parse_args()
    source = args.colabdesign_source or os.environ.get("COLABDESIGN_SOURCE")
    if source:
        sys.path.insert(0, str(Path(source).resolve()))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    case = json.loads(Path(args.case).read_text())
    settings = json.loads(Path(args.settings).read_text())
    backend = Backend(
        case,
        args.params,
        settings,
        (
            [f"model_{i}_multimer_v3" for i in args.models.split(",")]
            if args.models
            else None
        ),
    )

    def respond(obj):
        print("DCFLOW_JSON\t" + json.dumps(obj, allow_nan=False), flush=True)

    respond(dict(ready=True))
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        command = request.pop("command")
        identity = request.pop("id", None)
        if command == "shutdown":
            break
        try:
            if command == "evaluate":
                result = backend.evaluate(**request)
            elif command == "screen":
                result = backend.screen(**request)
            elif command == "refine":
                result = backend.refine(**request)
            else:
                raise ValueError("Unknown worker command")
            respond(dict(id=identity, results=result))
        except Exception as error:
            import traceback

            traceback.print_exc(file=sys.stderr)
            respond(dict(id=identity, error=f"{type(error).__name__}: {error}"))


if __name__ == "__main__":
    main()
