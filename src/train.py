"""Train FNO surrogate on TRL simulation data."""

import argparse
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fno_trl2d.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Loaded config: {cfg}")

    # TODO: build dataset, model, optimizer, training loop


if __name__ == "__main__":
    main()
