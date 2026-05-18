# Project Overview

## Goal
Build a fast surrogate model for turbulent radiative layer (TRL) simulations using Fourier Neural Operators (FNO).

## Motivation
High-fidelity TRL simulations are computationally expensive. A trained surrogate can replace expensive PDE solvers at inference time while maintaining acceptable accuracy.

## Success Criteria
- [ ] Train FNO on TRL simulation data
- [ ] Evaluate against held-out ground truth
- [ ] Publish model card on HuggingFace
