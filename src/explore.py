# explore.py
# Dataset exploration script for turbulent_radiative_layer_2D
from the_well.data import WellDataset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch

trainset = WellDataset(
    well_base_path='hf://datasets/polymathic-ai/',
    well_dataset_name='turbulent_radiative_layer_2D',
    well_split_name='train',
)

sample = trainset[0]
inp = sample['input_fields'][0]  # shape: [128, 384, 4]

field_names = ['Density', 'Pressure', 'Velocity X', 'Velocity Y']

fig, axes = plt.subplots(4, 1, figsize=(12, 10))
for i, (ax, name) in enumerate(zip(axes, field_names)):
    im = ax.imshow(inp[:, :, i].numpy(), cmap='viridis', aspect='auto')
    ax.set_title(name)
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('fluid_visualization.png', dpi=150)
print('Saved to fluid_visualization.png')