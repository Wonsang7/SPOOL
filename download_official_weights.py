"""Download the two author-provided FPFLI checkpoints used by the adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

import gdown


FILES = {
    "LLE_parameter_1_3ns.pth": "1SBhKLrbIfjm5C04GtBAB1ZdfarQuGKlh",
    "NIII_parameter_L8.pth.tar": "1BRow6g6EA7_JqpUbotW0GWi7zWCo2NZS",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", help="Path to a cloned Dooongx/FPFLI repository")
    args = parser.parse_args()
    target = Path(args.repo) / "Evaluation" / "model_parameters"
    target.mkdir(parents=True, exist_ok=True)
    for name, file_id in FILES.items():
        path = target / name
        if path.exists():
            print(f"exists: {path}")
            continue
        print(f"downloading: {path}")
        gdown.download(id=file_id, output=str(path), quiet=False)


if __name__ == "__main__":
    main()

