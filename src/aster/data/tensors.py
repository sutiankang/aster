"""Tensor-only local datasets, including nested multimodal records."""

from pathlib import Path
import torch
from ..core import digest_json, file_digest


class TensorTreeDataset:
    def __init__(self, path, *, preprocessing):
        self.path = Path(path)
        if not preprocessing or not isinstance(preprocessing, dict):
            raise ValueError("Tensor data requires explicit preprocessing/version/units metadata")
        self.file_fingerprint = file_digest(self.path)
        self.fingerprint = digest_json(
            {"file": self.file_fingerprint, "preprocessing": preprocessing}
        )
        self.tree = torch.load(self.path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(self.tree, dict) or not self.tree:
            raise ValueError("Tensor dataset must be a nonempty named tensor tree")
        lengths = []

        def check(node):
            if isinstance(node, dict) and node and all(isinstance(key, str) for key in node):
                for value in node.values():
                    check(value)
            elif isinstance(node, torch.Tensor) and node.ndim >= 1 and len(node) > 0:
                lengths.append(len(node))
            else:
                raise ValueError("All tensor-tree leaves must have a nonempty sample axis")

        check(self.tree)
        if len(set(lengths)) != 1:
            raise ValueError("Tensor-tree leaves have inconsistent sample counts")
        self.length = lengths[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)

        def select(node):
            return (
                {key: select(value) for key, value in node.items()}
                if isinstance(node, dict)
                else node[index]
            )

        return select(self.tree)

    def verify(self):
        if file_digest(self.path) != self.file_fingerprint:
            raise ValueError("Tensor dataset changed after indexing/checkpoint")
