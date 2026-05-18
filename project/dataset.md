# Dataset: turbulent_radiative_layer_2D

## What it is
Simulation of hot and cold gas layers moving past each 
other. The shear between them creates Kelvin-Helmholtz 
instability — the same instability you see when wind 
blows over water and creates waves. The mixing between 
hot and cold gas causes rapid cooling.

## Why it matters
This process happens in the interstellar medium and 
circumgalactic medium — when cold gas clouds move through 
hot ambient gas around galaxies. Understanding how mass 
transfers from hot to cold phase is important for galaxy 
formation models.

## Numbers
- Total size: 6.9GB
- Trajectories: 90 (10 random seeds × 9 cooling times)
- Timesteps per trajectory: 101
- Resolution: 384 × 128 pixels
- Fields: density, pressure, velocity x, velocity y
- Cooling times tested: 0.03, 0.06, 0.1, 0.18, 0.32, 
  0.56, 1.00, 1.78, 3.16

## Benchmark scores (lower is better)
- FNO: 0.5001
- TFNO: 0.5016  
- U-Net: 0.2418
- CNextU-Net: 0.1956 (best)

## What the model needs to learn
Given the current state of the gas (density, pressure, 
velocity at every grid point), predict the next timestep.

## HuggingFace location
polymathic-ai/turbulent_radiative_layer_2D

## Citation
Fielding et al. 2020, ApJL 894 L24