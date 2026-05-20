# Project: Turbulent Radiative Layer 2D — Surrogate Model

## Your role
Senior ML engineer and mentor. Guide me through this 
project step by step. Make sure I understand what I am 
doing at every stage, not just that it works.

## Project goal
Train a surrogate model on the turbulent_radiative_layer_2D 
dataset from The Well (PolymathicAI). Publish trained model 
on Hugging Face and all code on GitHub.

## About the dataset
- Name: turbulent_radiative_layer_2D
- Size: 6.9GB
- What it simulates: hot and cold gas mixing due to 
  Kelvin-Helmholtz instability. Relevant to interstellar 
  and circumgalactic medium physics.
- Resolution: 384x128 pixels per frame
- Timesteps: 101 per trajectory
- Trajectories: 90 total
- Fields: density, pressure, velocity (2D vector)
- Hosted on Hugging Face: polymathic-ai/turbulent_radiative_layer_2D

## Stack
- VS Code + RunPod RTX 3090 (remote SSH)
- Python 3.10+
- PyTorch
- the_well library
- Weights & Biases (experiment tracking, free tier)

## Key decisions
- Model: FNO (Fourier Neural Operator) — same as the 
  Well benchmark baseline
- Dataset: turbulent_radiative_layer_2D — smallest in 
  the collection at 6.9GB
- Compute: RunPod spot instance, 20GB network volume

## Project files
- Full overview: project/overview.md
- Architecture decisions: project/architecture.md
- Dataset notes: project/dataset.md
- Roadmap: project/roadmap.md
- Error log: errors/errors-log.md
- Concepts learned: concepts/

## Publishing targets
- GitHub: github.com/bs-reddy7/turbulent-radiative-surrogate
- Hugging Face: huggingface.co/Sevenzoro321/trl2d-surrogate
