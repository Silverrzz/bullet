#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def permute_dpbusd(net, l1, l2, offset):
    size = l1 * l2
    weights = np.frombuffer(net, dtype=np.int8, count=size, offset=offset)
    weights = weights.reshape(l1 // 4, 4, l2).swapaxes(1, 2)
    return net[:offset] + weights.tobytes() + net[offset + size:]


def unpermute_dpbusd(net, l1, l2, offset):
    size = l1 * l2
    weights = np.frombuffer(net, dtype=np.int8, count=size, offset=offset)
    weights = weights.reshape(l1 // 4, l2, 4).swapaxes(1, 2)
    return net[:offset] + weights.tobytes() + net[offset + size:]


def perm_sparse(net, permutation, l1, l2, buckets):
    l0w_size = 2 * buckets * 843 * l1
    l0b_size = 2 * l1
    l1_offset = l0w_size + l0b_size
    l1w_size = l1 * l2
    l0w = np.frombuffer(net, dtype="<i2", count=buckets * 843 * l1)
    l0w = l0w.reshape(buckets, 843, l1)
    l0b = np.frombuffer(net, dtype="<i2", count=l1, offset=l0w_size)
    l1w = np.frombuffer(net, dtype=np.int8, count=l1w_size, offset=l1_offset)
    l1w = l1w.reshape(l1, l2)
    return (
        l0w[:, :, permutation].tobytes()
        + l0b[permutation].tobytes()
        + l1w[permutation, :].tobytes()
        + net[l1_offset + l1w_size:]
    )


def main():
    parser = argparse.ArgumentParser(description="Sparse-permute a dpbusd-permuted wakwak net.")
    parser.add_argument("net", type=Path)
    parser.add_argument("--permutation", type=Path, default=Path("perm.json"))
    parser.add_argument("--output", type=Path, default=Path("permuted.nnue"))
    parser.add_argument("--l1-size", type=int, default=1024)
    parser.add_argument("--l2-size", type=int)
    parser.add_argument("--buckets", type=int, default=1)
    args = parser.parse_args()
    l1 = args.l1_size
    if l1 <= 0 or l1 % 4 or args.buckets <= 0:
        parser.error("L1 must be a positive multiple of four; buckets must be positive")
    net = args.net.read_bytes()
    offset = 2 * (args.buckets * 843 + 1) * l1
    candidates = [args.l2_size] if args.l2_size is not None else [16, 32]
    matching = []
    for l2 in candidates:
        if l2 <= 0:
            parser.error("L2 must be positive")
        size = offset + l1 * l2 + 4 * l2 + 4 * 2 * (l2 - 1) * 32 + 4 * 32 + 4 * 32 + 4
        if len(net) == (size + 63) // 64 * 64:
            matching.append(l2)
    if len(matching) != 1:
        parser.error("Network size does not match a supported 1024/16-or-32/32 layout")
    l2 = matching[0]
    perm = np.asarray(json.loads(args.permutation.read_text()))
    if (
        perm.ndim != 1
        or perm.dtype.kind not in "iu"
        or len(perm) != l1 // 2
        or not np.array_equal(np.sort(perm), np.arange(l1 // 2))
    ):
        parser.error("Permutation must contain each paired FT neuron index exactly once")
    perm2 = np.concatenate((perm, perm + len(perm)))
    logical = unpermute_dpbusd(net, l1, l2, offset)
    reordered = perm_sparse(logical, perm2, l1, l2, args.buckets)
    output = permute_dpbusd(reordered, l1, l2, offset)
    recovered = perm_sparse(reordered, np.argsort(perm2), l1, l2, args.buckets)
    if permute_dpbusd(recovered, l1, l2, offset) != net:
        raise ValueError("Inverse permutation failed to recover the original network")
    with args.output.open("xb") as destination:
        destination.write(output)
    print(f"Wrote {args.output}: {l1}->{l2}->32, {len(output)} bytes")


if __name__ == "__main__":
    main()
