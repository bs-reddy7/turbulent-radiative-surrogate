# What Is a Fourier Neural Operator (FNO)?

## Definition
A Fourier Neural Operator (FNO) is a neural network architecture designed to learn mappings between function spaces, particularly useful for solving PDEs.

## Core Idea
Instead of learning pointwise operations, FNO learns in the frequency domain:
1. Apply FFT to the input field
2. Filter the lowest `k` Fourier modes with learned weights
3. Apply inverse FFT
4. Add a residual linear transform (applied in physical space)

## Advantages for PDEs
- Resolution-invariant: can train at one resolution and evaluate at another
- Efficiently captures global structure via spectral operations
- Strong empirical results on Navier-Stokes, Darcy flow, etc.

## Key Paper
Li et al. (2021) — "Fourier Neural Operator for Parametric Partial Differential Equations"  
https://arxiv.org/abs/2010.08895
