"""Inject an existing receptor A3M into AfDesign binder inputs."""

import string
import jax.numpy as jnp
import numpy as np
from Bio.Align import PairwiseAligner

AF_ORDER = "ARNDCQEGHILKMFPSTWYVX-"
AA_TO_INDEX = {amino_acid: index for (index, amino_acid) in enumerate(AF_ORDER)}


def parse_a3m(path, limit):
    sequences = []
    deletions = []
    sequence = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith(">"):
            if sequence:
                (aligned, deletion) = aligned_sequence("".join(sequence))
                sequences.append(aligned)
                deletions.append(deletion)
                sequence = []
                if len(sequences) >= limit:
                    break
        elif line.strip():
            sequence.append(line.strip())
    if sequence and len(sequences) < limit:
        (aligned, deletion) = aligned_sequence("".join(sequence))
        sequences.append(aligned)
        deletions.append(deletion)
    if not sequences:
        raise ValueError(f"No sequences in A3M: {path}")
    length = len(sequences[0])
    if any((len(item) != length for item in sequences)):
        raise ValueError(f"Inconsistent aligned lengths in A3M: {path}")
    return (sequences, np.asarray(deletions, dtype=np.float32))


def aligned_sequence(sequence):
    aligned = []
    deletion = []
    insertion_length = 0
    for amino_acid in sequence:
        if amino_acid in string.ascii_lowercase:
            insertion_length += 1
        else:
            aligned.append(amino_acid.upper())
            deletion.append(insertion_length)
            insertion_length = 0
    return ("".join(aligned), deletion)


class ReceptorMSA:
    """Mutable callback so one loaded AfDesign model can serve multiple targets."""

    def __init__(self, num_sequences):
        if num_sequences < 1:
            raise ValueError("num_sequences must be positive")
        self.num_sequences = num_sequences
        self.features = None
        self.mask = None
        self.query = None
        self.sequences = None
        self.deletions = None
        self.binder_length = None

    def load(self, path, binder_length):
        (sequences, deletions) = parse_a3m(path, self.num_sequences)
        self.query = sequences[0]
        self.sequences = sequences
        self.deletions = deletions
        self.binder_length = binder_length

    def _project(self, receptor_sequence):
        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 2.0
        aligner.mismatch_score = -1.0
        aligner.open_gap_score = -5.0
        aligner.extend_gap_score = -0.5
        alignment = aligner.align(self.query, receptor_sequence)[0]
        (query_blocks, receptor_blocks) = alignment.aligned
        projection = np.full(len(receptor_sequence), -1, dtype=int)
        for (query_start, query_end), (receptor_start, receptor_end) in zip(
            query_blocks, receptor_blocks
        ):
            block_length = min(query_end - query_start, receptor_end - receptor_start)
            projection[receptor_start : receptor_start + block_length] = np.arange(
                query_start, query_start + block_length
            )
        if np.any(projection < 0):
            missing = np.where(projection < 0)[0].tolist()
            raise ValueError(f"A3M query cannot map receptor positions: {missing[:10]}")
        projected_query = "".join((self.query[index] for index in projection))
        identity = sum(
            (a == b for (a, b) in zip(projected_query, receptor_sequence))
        ) / len(receptor_sequence)
        if identity < 0.95:
            raise ValueError(f"A3M query/receptor identity is only {identity:.3f}")
        rows = [
            "".join((sequence[index] for index in projection))
            for sequence in self.sequences[1:]
        ]
        row_deletions = self.deletions[1:, projection]
        receptor_length = len(receptor_sequence)
        binder_length = self.binder_length
        feature_rows = np.zeros(
            (self.num_sequences - 1, receptor_length + binder_length, 49)
        )
        mask_rows = np.zeros((self.num_sequences - 1, receptor_length + binder_length))
        for row_index, (sequence, deletion) in enumerate(zip(rows, row_deletions)):
            indices = np.asarray(
                [AA_TO_INDEX.get(amino_acid, 20) for amino_acid in sequence]
            )
            one_hot = np.eye(22, dtype=np.float32)[indices]
            feature_rows[row_index, :receptor_length, :22] = one_hot
            feature_rows[row_index, :receptor_length, 23] = deletion > 0
            deletion_value = np.arctan(deletion / 3.0) * (2.0 / np.pi)
            feature_rows[row_index, :receptor_length, 24] = deletion_value
            feature_rows[row_index, :receptor_length, 25:47] = one_hot
            feature_rows[row_index, :receptor_length, 48] = deletion_value
            feature_rows[row_index, receptor_length:, 21] = 1.0
            feature_rows[row_index, receptor_length:, 46] = 1.0
            mask_rows[row_index] = 1.0
        self.features = jnp.asarray(feature_rows)
        self.mask = jnp.asarray(mask_rows)

    def validate_query(self, model):
        from colabdesign.af.alphafold.common import residue_constants

        indices = np.asarray(model._pdb["batch"]["aatype"][: model._target_len])
        receptor_sequence = "".join(
            (residue_constants.restypes_with_x[int(index)] for index in indices)
        )
        if receptor_sequence != self.query:
            raise ValueError(
                f"A3M query does not match normalized receptor sequence: {self.query[:30]}... != {receptor_sequence[:30]}..."
            )

    def install(self, model):
        """Resize MSA inputs while keeping one optimized/predicted binder sequence."""
        from colabdesign.af.alphafold.common import residue_constants

        indices = np.asarray(model._pdb["batch"]["aatype"][: model._target_len])
        receptor_sequence = "".join(
            (residue_constants.restypes_with_x[int(index)] for index in indices)
        )
        self._project(receptor_sequence)
        length = model._target_len + model._binder_len
        model._inputs["msa_feat"] = np.zeros((self.num_sequences, length, 49))
        model._inputs["msa_mask"] = np.ones((self.num_sequences, length))
        model._inputs["msa_row_mask"] = np.ones(self.num_sequences)
        model._inputs["receptor_msa_feat"] = self.features
        model._inputs["receptor_msa_mask"] = self.mask

    def callback(self, inputs):
        inputs["msa_feat"] = inputs["msa_feat"].at[1:].set(inputs["receptor_msa_feat"])
        inputs["msa_mask"] = inputs["msa_mask"].at[1:].set(inputs["receptor_msa_mask"])
        inputs["msa_row_mask"] = (
            inputs["msa_row_mask"]
            .at[1:]
            .set(jnp.any(inputs["receptor_msa_mask"], axis=1))
        )
