"""Visualize predictions vs ground truth for TRL surrogate."""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()

    # TODO: load model and data, plot side-by-side fields, save figure


if __name__ == "__main__":
    main()
