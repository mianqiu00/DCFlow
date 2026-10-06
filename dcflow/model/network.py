import torch
from torch import nn
from dcflow.model.node import NodeEmbedder
from dcflow.model.edge import EdgeEmbedder
from dcflow.model.ga import GAEncoder
from dcflow.model.cyclic import CyclicConditioning
from dcflow.model.mpnn_prior import ProteinMPNNPrior
from dcflow.geometry.protein.constants import BBHeavyAtom, max_num_heavyatoms
from dcflow.geometry.common.geometry import construct_3d_basis
from dcflow.geometry.common.layers import clampped_one_hot


class DCFlow(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self._model_cfg = cfg.encoder
        self._interpolant_cfg = cfg.interpolant
        self.node_embedder = NodeEmbedder(
            cfg.encoder.node_embed_size, max_num_heavyatoms
        )
        self.edge_embedder = EdgeEmbedder(
            cfg.encoder.edge_embed_size, max_num_heavyatoms
        )
        self.ga_encoder = GAEncoder(cfg.encoder.ipa)
        cyclic_cfg = cfg.get("cyclic", {})
        self.cyclic_conditioning = (
            CyclicConditioning(
                cfg.encoder.node_embed_size,
                cfg.encoder.edge_embed_size,
                int(cyclic_cfg.get("max_relpos", 32)),
            )
            if cyclic_cfg.get("enabled", False)
            else None
        )
        mpnn_cfg = cfg.get("mpnn_prior", {})
        self.mpnn_prior = (
            ProteinMPNNPrior(mpnn_cfg) if mpnn_cfg.get("enabled", False) else None
        )
        self.mpnn_gate = (
            nn.Parameter(torch.tensor(float(mpnn_cfg.get("fusion_init", 0.0))))
            if self.mpnn_prior is not None
            else None
        )
        self.sample_structure = self._interpolant_cfg.sample_structure
        self.sample_sequence = self._interpolant_cfg.sample_sequence
        self.K = self._interpolant_cfg.seqs.num_classes
        self.k = self._interpolant_cfg.seqs.simplex_value
        self.sequence_start_t = float(
            self._interpolant_cfg.get("sequence_start_t", 0.0)
        )
        self.sidechain_start_t = float(
            self._interpolant_cfg.get("sidechain_start_t", 0.0)
        )

    @staticmethod
    def _clock(t, start_t):
        if start_t <= 0:
            return t
        return ((t - start_t) / (1.0 - start_t)).clamp(0.0, 1.0)

    def _condition(self, node_embed, edge_embed, batch):
        if self.cyclic_conditioning is not None:
            (node_embed, edge_embed) = self.cyclic_conditioning(
                node_embed, edge_embed, batch
            )
        return (node_embed, edge_embed)

    def _sequence_logits(self, logits, pred_trans, pred_rotmats, batch, seq_t, angles):
        if self.mpnn_prior is not None:
            prior = self.mpnn_prior(
                pred_trans,
                pred_rotmats,
                batch["res_mask"],
                batch["chain_nb"],
                batch["res_nb"],
                angles,
                batch,
            )
            prior = prior - prior.mean(dim=-1, keepdim=True)
            logits = logits + (seq_t > 0)[..., None] * self.mpnn_gate * prior
        if "allowed_aa_mask" in batch:
            logits = logits.masked_fill(batch["allowed_aa_mask"] <= 0, -1000000000.0)
        return logits

    def encode(self, batch):
        rotmats_1 = construct_3d_basis(
            batch["pos_heavyatom"][:, :, BBHeavyAtom.CA],
            batch["pos_heavyatom"][:, :, BBHeavyAtom.C],
            batch["pos_heavyatom"][:, :, BBHeavyAtom.N],
        )
        trans_1 = batch["pos_heavyatom"][:, :, BBHeavyAtom.CA]
        seqs_1 = batch["aa"]
        angles_1 = batch["torsion_angle"]
        context_mask = torch.logical_and(
            batch["mask_heavyatom"][:, :, BBHeavyAtom.CA], ~batch["generate_mask"]
        )
        structure_mask = context_mask if self.sample_structure else None
        sequence_mask = context_mask if self.sample_sequence else None
        node_embed = self.node_embedder(
            batch["aa"],
            batch["res_nb"],
            batch["chain_nb"],
            batch["pos_heavyatom"],
            batch["mask_heavyatom"],
            structure_mask=structure_mask,
            sequence_mask=sequence_mask,
        )
        edge_embed = self.edge_embedder(
            batch["aa"],
            batch["res_nb"],
            batch["chain_nb"],
            batch["pos_heavyatom"],
            batch["mask_heavyatom"],
            structure_mask=structure_mask,
            sequence_mask=sequence_mask,
        )
        return (rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed)

    def zero_center_part(self, pos, gen_mask, res_mask):
        """
        move pos by center of gen_mask
        pos: (B,N,3)
        gen_mask, res_mask: (B,N)
        """
        center = torch.sum(pos * gen_mask[..., None], dim=1) / (
            torch.sum(gen_mask, dim=-1, keepdim=True) + 1e-08
        )
        center = center.unsqueeze(1)
        pos = pos - center
        pos = pos * res_mask[..., None]
        return (pos, center)

    def seq_to_simplex(self, seqs):
        return clampped_one_hot(seqs, self.K).float() * self.k * 2 - self.k
