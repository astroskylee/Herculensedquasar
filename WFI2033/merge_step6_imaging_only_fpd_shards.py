#!/usr/bin/env python3
from __future__ import annotations

import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import glob
import re
from pathlib import Path

import arviz as az


INPUT_PATH = Path(
    "/mnt/lustre/tianli/quasar_hmc/"
    "WFI2033_ss=2_inferh0_step6_imaging_only_20260413_19/"
    "WFI2033_all_ss=2_inferh0_step6_imaging_only.nc"
)
DEFAULT_OUTPUT_PATH = INPUT_PATH.with_name("WFI2033_all_ss=2_inferh0_step6_imaging_only_withFPD.nc")


def default_shard_glob(input_path):
    return input_path.with_name(f"{input_path.stem}_withFPD_chain*.nc")


def shard_index(path):
    match = re.search(r"_chain(\d+)\.nc$", path.name)
    if match is None:
        raise ValueError(f"Could not parse chain index from shard name: {path}")
    return int(match.group(1))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--shard-glob", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    shard_pattern = args.shard_glob if args.shard_glob is not None else str(default_shard_glob(args.input))
    shard_paths = sorted([Path(path) for path in glob.glob(shard_pattern)], key=shard_index)
    if not shard_paths:
        raise FileNotFoundError(f"No shard files matched: {shard_pattern}")

    merged = az.from_netcdf(shard_paths[0])
    for shard_path in shard_paths[1:]:
        merged = az.concat(merged, az.from_netcdf(shard_path), dim="chain")

    merged.to_netcdf(args.output)
    print(f"Merged {len(shard_paths)} shards into: {args.output}")


if __name__ == "__main__":
    main()
