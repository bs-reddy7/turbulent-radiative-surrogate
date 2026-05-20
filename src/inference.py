import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from the_well.data import WellDataset
from the_well.benchmark.models import FNO

testset = WellDataset(
    well_base_path='/workspace/data/datasets/',
    well_dataset_name='turbulent_radiative_layer_2D',
    well_split_name='test',
    n_steps_input=4,
    n_steps_output=1,
)

sample = testset[0]
# input_fields: [T=4, H, W, C=4] → [1, T, H, W, C] → [1, T*C=16, H, W]
inp = sample['input_fields'].unsqueeze(0).cuda()    # [1, 4, 128, 384, 4]
target = sample['output_fields'].unsqueeze(0).cuda() # [1, 1, 128, 384, 4]
B, T, H, W, C = inp.shape
inp_cf = inp.permute(0, 1, 4, 2, 3).reshape(B, T * C, H, W)  # [1, 16, 128, 384]
target_cf = target.squeeze(1).permute(0, 3, 1, 2)            # [1, 4, 128, 384]

checkpoint = torch.load(
    '/workspace/the_well/the_well/benchmark/experiments/turbulent_radiative_layer_2D-fno-FNO-0.001/0/checkpoints/best.pt',
    map_location='cuda'
)

model = FNO(
    dim_in=16,
    dim_out=4,
    n_spatial_dims=2,
    spatial_resolution=(128, 384),
    modes1=16,
    modes2=16,
    hidden_channels=128,
).cuda()

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print('Model loaded successfully!')

with torch.no_grad():
    pred = model(inp_cf)

print('Prediction shape:', pred.shape)

field_names = ['Density', 'Pressure', 'Velocity X', 'Velocity Y']
fig, axes = plt.subplots(4, 4, figsize=(18, 12))
fig.suptitle('FNO: Input → Prediction vs Ground Truth', fontsize=14)

for i, name in enumerate(field_names):
    # channels 12-15 = most recent (4th) input time step's 4 fields
    inp_field = inp_cf[0, 12 + i].cpu().numpy()
    pred_field = pred[0, i].cpu().numpy()
    true_field = target_cf[0, i].cpu().numpy()
    error = abs(pred_field - true_field)

    axes[i, 0].imshow(inp_field, cmap='viridis', aspect='auto')
    axes[i, 0].set_title(f'{name} - Input (last step)')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(pred_field, cmap='viridis', aspect='auto')
    axes[i, 1].set_title(f'{name} - Predicted')
    axes[i, 1].axis('off')

    axes[i, 2].imshow(true_field, cmap='viridis', aspect='auto')
    axes[i, 2].set_title(f'{name} - Ground Truth')
    axes[i, 2].axis('off')

    axes[i, 3].imshow(error, cmap='hot', aspect='auto')
    axes[i, 3].set_title(f'{name} - Error')
    axes[i, 3].axis('off')

plt.tight_layout()
plt.savefig('/workspace/turbulent-radiative-surrogate/model_predictions.png', dpi=150)
print('Saved!')
