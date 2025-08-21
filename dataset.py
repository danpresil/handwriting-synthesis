import os
import numpy as np
import torch
from torch.utils.data import Dataset


class HandwritingDataset(Dataset):
    """Dataset for handwriting synthesis data stored in numpy arrays.

    The directory is expected to contain the following files:
        - x.npy: stroke sequences of shape (N, T, 3)
        - x_len.npy: lengths of stroke sequences
        - c.npy: character sequences
        - c_len.npy: lengths of character sequences
    """

    def __init__(self, data_dir):
        self.x = np.load(os.path.join(data_dir, 'x.npy'))
        self.x_len = np.load(os.path.join(data_dir, 'x_len.npy'))
        self.c = np.load(os.path.join(data_dir, 'c.npy'))
        self.c_len = np.load(os.path.join(data_dir, 'c_len.npy'))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_full = self.x[idx]
        x_len = int(self.x_len[idx]) - 1
        c_len = int(self.c_len[idx])

        x = x_full[:x_len]
        y = x_full[1:x_len + 1]
        c = self.c[idx][:c_len]

        return (
            x.astype(np.float32),
            y.astype(np.float32),
            np.int32(x_len),
            c.astype(np.int32),
            np.int32(c_len),
        )


def handwriting_collate_fn(batch):
    """Pad variable length sequences and convert to tensors."""
    xs, ys, x_lens, cs, c_lens = zip(*batch)

    batch_size = len(xs)
    max_x_len = max(x_lens)
    max_c_len = max(c_lens)

    x_dim = xs[0].shape[1]
    x_batch = np.zeros((batch_size, max_x_len, x_dim), dtype=np.float32)
    y_batch = np.zeros((batch_size, max_x_len, x_dim), dtype=np.float32)
    c_batch = np.zeros((batch_size, max_c_len), dtype=np.int32)

    for i in range(batch_size):
        x_batch[i, : x_lens[i]] = xs[i]
        y_batch[i, : x_lens[i]] = ys[i]
        c_batch[i, : c_lens[i]] = cs[i]

    return (
        torch.from_numpy(x_batch),
        torch.from_numpy(y_batch),
        torch.from_numpy(np.asarray(x_lens, dtype=np.int32)),
        torch.from_numpy(c_batch),
        torch.from_numpy(np.asarray(c_lens, dtype=np.int32)),
    )
