#!/usr/bin/env python3
"""Analyze WISQ JSON files: count steps, total IDs, and average IDs per step."""

import json
import sys
import os


def analyze_wisq_json(filepath):
    print(f"File: {filepath}")
    with open(filepath) as f:
        data = json.load(f)

    steps = data["steps"]
    num_steps = len(steps)

    all_ids = [entry["id"] for step in steps for entry in step]
    num_ids = len(all_ids)

    ids_per_step = [len(step) for step in steps]
    avg_ids_per_step = num_ids / num_steps if num_steps > 0 else 0

    print(f"  Number of steps:              {num_steps}")
    print(f"  Total number of IDs:          {num_ids}")
    print(f"  Average IDs per step:         {avg_ids_per_step:.2f}")
    print()


def main():
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        # Default: all JSON files in results/all_wisq
        directory = os.path.join(os.path.dirname(__file__), "..", "results", "all_wisq")
        files = sorted(
            os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".json")
        )

    for filepath in files:
        analyze_wisq_json(filepath)


if __name__ == "__main__":
    main()
