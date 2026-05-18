# What Is a Surrogate Model?

## Definition
A surrogate model (also called an emulator or proxy model) is a cheap-to-evaluate approximation of an expensive computational model.

## When to Use
- When the true simulator takes hours/days per run
- When you need many evaluations (optimization, uncertainty quantification, real-time inference)

## Types
| Type | Example |
|------|---------|
| Statistical | Gaussian Process |
| Neural | FNO, U-Net, DeepONet |
| Reduced-order | POD-based ROMs |

## Evaluation Metrics
- **Relative L2 error** between surrogate output and ground truth
- **Rollout stability**: does error accumulate over long rollouts?
- **Inference speedup**: how much faster than the true solver?
