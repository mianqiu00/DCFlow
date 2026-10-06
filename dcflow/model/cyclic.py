import heapq
import torch
from torch import nn
from dcflow.geometry.protein.constants import AA, BBHeavyAtom

TOPOLOGY_MODE = {"Linear": 0, "HeadToTail": 1, "CysToCys": 2}
CLOSURE_TYPE = {"none": 0, "HeadToTail": 1, "CysToCys": 2}


def attach_cyclic_features(data, topology_mode, terminal_ss=False):
    """Attach cyclic graph features to a receptor-conditioned design input."""
    num_res = data["aa"].shape[0]
    peptide_idx = torch.where(data["generate_mask"])[0]
    closure_endpoints = torch.full((num_res, 2), -1, dtype=torch.long)
    closure_type = torch.zeros(num_res, dtype=torch.long)
    cyclization_site = torch.zeros(num_res, dtype=torch.bool)
    allowed_aa = torch.ones(num_res, 20)
    closures = []
    if topology_mode == "HeadToTail" and len(peptide_idx) >= 2:
        left, right = (int(peptide_idx[0]), int(peptide_idx[-1]))
        closures.append((left, right, CLOSURE_TYPE[topology_mode]))
    elif topology_mode == "CysToCys" and terminal_ss:
        left, right = (int(peptide_idx[0]), int(peptide_idx[-1]))
        closures.append((left, right, CLOSURE_TYPE[topology_mode]))
        allowed_aa[[left, right]] = 0
        allowed_aa[[left, right], int(AA.CYS)] = 1
    elif topology_mode == "CysToCys":
        cys_idx = peptide_idx[data["aa"][peptide_idx] == int(AA.CYS)]
        if len(cys_idx) >= 2:
            sg = data["pos_heavyatom"][cys_idx, 5]
            sg_mask = data["mask_heavyatom"][cys_idx, 5]
            distances = torch.cdist(sg, sg)
            valid = sg_mask[:, None] & sg_mask[None, :]
            valid &= ~torch.eye(len(cys_idx), dtype=torch.bool)
            distances = distances.masked_fill(~valid, float("inf"))
            flat_idx = torch.argmin(distances)
            i = int(flat_idx // len(cys_idx))
            j = int(flat_idx % len(cys_idx))
            if torch.isfinite(distances[i, j]):
                left, right = (int(cys_idx[i]), int(cys_idx[j]))
                closures.append((left, right, CLOSURE_TYPE[topology_mode]))
                allowed_aa[left] = 0
                allowed_aa[right] = 0
                allowed_aa[left, int(AA.CYS)] = 1
                allowed_aa[right, int(AA.CYS)] = 1
    adjacency = [[] for _ in range(num_res)]
    peptide_set = set(peptide_idx.tolist())
    for i in range(num_res - 1):
        if i in peptide_set and i + 1 in peptide_set:
            adjacency[i].append((i + 1, 0))
            adjacency[i + 1].append((i, 0))
    direct_closure = torch.zeros(num_res, num_res, dtype=torch.long)
    for left, right, kind in closures:
        adjacency[left].append((right, 1))
        adjacency[right].append((left, 1))
        direct_closure[left, right] = kind
        direct_closure[right, left] = kind
        closure_endpoints[left] = torch.tensor([left, right])
        closure_type[left] = kind
        cyclization_site[left] = True
        cyclization_site[right] = True
    disconnected = num_res + 1
    graph_relpos = torch.full((num_res, num_res), disconnected, dtype=torch.long)
    path_type = torch.full((num_res, num_res), 2, dtype=torch.long)
    for src in peptide_idx.tolist():
        dist = [10**9] * num_res
        used_closure = [2] * num_res
        dist[src], used_closure[src] = (0, 0)
        heap = [(0, 0, src)]
        while heap:
            current_dist, current_type, node = heapq.heappop(heap)
            if (current_dist, current_type) > (dist[node], used_closure[node]):
                continue
            for neighbor, edge_type in adjacency[node]:
                candidate = (current_dist + 1, max(current_type, edge_type))
                if candidate < (dist[neighbor], used_closure[neighbor]):
                    dist[neighbor], used_closure[neighbor] = candidate
                    heapq.heappush(heap, (*candidate, neighbor))
        for dst in peptide_idx.tolist():
            if dist[dst] < 10**9:
                graph_relpos[src, dst] = dist[dst]
                path_type[src, dst] = used_closure[dst]
    pair_mask = data["generate_mask"][:, None] & data["generate_mask"][None, :]
    data.update(
        {
            "topology_mode_id": torch.full(
                (num_res,), TOPOLOGY_MODE.get(topology_mode, 0), dtype=torch.long
            ),
            "cyclic_graph_relpos": graph_relpos,
            "cyclic_path_type": path_type,
            "cyclic_direct_closure": direct_closure,
            "topology_pair_mask": pair_mask,
            "closure_endpoints": closure_endpoints,
            "closure_type": closure_type,
            "cyclization_site_mask": cyclization_site,
            "allowed_aa_mask": allowed_aa,
        }
    )
    return data


class CyclicConditioning(nn.Module):

    def __init__(self, node_dim, edge_dim, max_relpos=32):
        super().__init__()
        self.max_relpos = max_relpos
        self.mode_embedding = nn.Embedding(3, node_dim)
        self.site_embedding = nn.Embedding(2, node_dim)
        self.relpos_embedding = nn.Embedding(max_relpos + 2, edge_dim)
        self.path_embedding = nn.Embedding(3, edge_dim)
        self.closure_embedding = nn.Embedding(3, edge_dim)
        self.node_gate = nn.Parameter(torch.zeros(()))
        self.edge_gate = nn.Parameter(torch.zeros(()))

    def forward(self, node_embed, edge_embed, batch):
        mode = batch["topology_mode_id"][:, 0].clamp(0, 2)
        site = batch["cyclization_site_mask"].long().clamp(0, 1)
        node_delta = self.mode_embedding(mode)[:, None] + self.site_embedding(site)
        relpos = batch["cyclic_graph_relpos"].clamp(0, self.max_relpos + 1)
        edge_delta = self.relpos_embedding(relpos)
        edge_delta = edge_delta + self.path_embedding(
            batch["cyclic_path_type"].clamp(0, 2)
        )
        edge_delta = edge_delta + self.closure_embedding(
            batch["cyclic_direct_closure"].clamp(0, 2)
        )
        edge_delta = edge_delta * batch["topology_pair_mask"][..., None]
        return (
            node_embed + self.node_gate * node_delta,
            edge_embed + self.edge_gate * edge_delta,
        )
