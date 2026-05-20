#!/bin/bash
# Training command used to train the FNO surrogate model
# Run from: /workspace/the_well/the_well/benchmark/

python train.py \
    experiment=fno \
    server=local \
    data=turbulent_radiative_layer_2D \
    data.well_base_path=/workspace/data/datasets/ \
    data.batch_size=4 \
    optimizer.lr=1e-3
