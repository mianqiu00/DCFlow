from pathlib import Path
import sys
import torch
from torch import nn

MPNN_TO_DCFLOW = list(range(20))


class ProteinMPNNPrior(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        source_dir = Path(cfg.source_dir).expanduser().resolve()
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        from protein_mpnn_utils import ProteinMPNN

        checkpoint = torch.load(
            Path(cfg.checkpoint_path).expanduser(),
            map_location="cpu",
            weights_only=False,
        )
        self.model = ProteinMPNN(
            ca_only=False,
            num_letters=21,
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            vocab=21,
            k_neighbors=int(checkpoint.get("num_edges", 48)),
            augment_eps=0.0,
            dropout=0.0,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.requires_grad_(False)
        self.register_buffer("index", torch.tensor(MPNN_TO_DCFLOW), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, trans, rotmats, res_mask, chain_nb, res_nb, angles, batch):
        with torch.no_grad(), torch.autocast(
            device_type=trans.device.type, enabled=False
        ):
            from dcflow.model.torsion import full_atom_reconstruction

            (atoms, _, _) = full_atom_reconstruction(
                rotmats.float(),
                trans.float(),
                angles.float(),
                torch.zeros_like(res_nb).long(),
            )
            coords = torch.where(
                batch["generate_mask"][..., None, None],
                atoms[:, :, :4],
                batch["pos_heavyatom"][:, :, :4].float(),
            )
            log_probs = self.model.unconditional_probs(
                coords, res_mask.float(), res_nb.long(), chain_nb.long()
            )
        return log_probs.index_select(-1, self.index)
