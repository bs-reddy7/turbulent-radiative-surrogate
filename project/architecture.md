# Architecture

## Model: Fourier Neural Operator (FNO)

### Key Components
- **Fourier Layer**: lifts input to spectral domain, filters modes, projects back
- **MLP bypass**: residual connection through a linear layer
- **Stacked blocks**: multiple FNO layers for depth

### Input / Output
- **Input**: initial condition fields (e.g., temperature, velocity) on a 2D grid
- **Output**: predicted field at the next timestep (or rollout)

## References
- Li et al. (2021) — "Fourier Neural Operator for Parametric Partial Differential Equations"
