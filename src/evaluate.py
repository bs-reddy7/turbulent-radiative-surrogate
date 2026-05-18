"""Evaluate trained FNO surrogate against ground-truth simulations."""

import argparse


def relative_l2(pred, target):
    return ((pred - target) ** 2).sum() ** 0.5 / (target ** 2).sum() ** 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    # TODO: load model, load data, run evaluation, print metrics


if __name__ == "__main__":
    main()
