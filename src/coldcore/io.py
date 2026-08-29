"""Solution file I/O (atomic writes; format = digit rows, space-separated).

Compatible with the upstream record repo's code files: one point per
line, n digits separated by spaces.
"""
import os

import numpy as np


def write_solution_atomic(path, words):
    """words: (M, n) digit array. Write via tmp+rename (crash-safe)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for row in np.asarray(words):
            f.write(" ".join(str(int(x)) for x in row) + "\n")
    os.replace(tmp, path)


def read_solution(path, q=None, n=None):
    """Read a digit-row file -> (M, n) uint8 array; validates against q, n."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([int(x) for x in line.split()])
    arr = np.array(rows, dtype=np.uint8)
    if n is not None and arr.shape[1] != n:
        raise ValueError(f"{path}: rows have {arr.shape[1]} digits, want n={n}")
    if q is not None and arr.size and arr.max() >= q:
        raise ValueError(f"{path}: digit {arr.max()} out of range for q={q}")
    return arr
