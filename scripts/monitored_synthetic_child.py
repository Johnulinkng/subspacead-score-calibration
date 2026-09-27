"""Short CPU-only child used exclusively by Invoke-MonitoredExperiment.ps1 -SelfTest."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--claim-eligible", choices=("true", "false"), required=True)
    parser.add_argument("--sentinel", required=True)
    args = parser.parse_args()

    if args.claim_eligible != "false":
        raise RuntimeError("The synthetic launcher self-test must be claim-ineligible.")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("The synthetic launcher self-test must hide all GPUs.")
    if args.sentinel != 'space and "quote" ending\\':
        raise RuntimeError("Windows command-line argument quoting changed the sentinel.")
    if not 0.0 <= args.duration_seconds <= 10.0:
        raise ValueError("Synthetic self-test duration must be between 0 and 10 seconds.")

    print(
        json.dumps(
            {
                "kind": "monitored_launcher_self_test",
                "claim_eligible": False,
                "gpu_visible": False,
                "pid": os.getpid(),
                "sentinel": args.sentinel,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print("synthetic self-test stderr sentinel", file=sys.stderr, flush=True)
    deadline = time.monotonic() + args.duration_seconds
    payload = bytearray(2 * 1024 * 1024)
    while time.monotonic() < deadline:
        payload[0] = (payload[0] + 1) % 255
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    print(json.dumps({"status": "completed", "claim_eligible": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
