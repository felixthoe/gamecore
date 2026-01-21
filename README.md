# gamecore — Core Library for Dynamic and Differential Games

## Description
A Python-based library for simulating Differential and Dynamic Games.
This package builds a shared core library of the research group "Cooperative Systems" at the Institute of Control Systems (IRS), Karlsruhe Institute of Technology (KIT).

## Installation
Install this package with 
```bash
pip install gamecore @ git+https://https://gitlab.kit.edu/kit/irs/rus/ks/gamecore.git@v0.1.0
```
and change the version number as required.

## Usage
### Getting Started
You can get to know this codebase with the tutorial notebooks in `notebooks/tutorials/`.

## Overall Project Structure

The `gamecore` package provides the following conceptual building blocks:

```text
gamecore
├── Game abstraction
│   ├── BaseGame                 # Holds BaseSystem and BasePlayer[]
│   ├── BaseSystem               # Defines the ODEs / system dynamics
│   ├── BasePlayer               # Holds a strategy and a cost function
│   ├── BaseStrategy             # Abstract interface for control laws
│   ├── BaseStrategyTrajectory   # Tracking and reconstruction of learned BaseStrategies and costs
│   └── BaseCost                 # Abstract interface for cost functions
│
├── Linear-Quadratic specialization
│   ├── LQGame                   # Specialization with LinearSystem and LQPlayer[]
│   ├── LinearSystem             # Linear version of BaseSystem
│   ├── LQPlayer                 # Player with LinearStrategy and QuadraticCost
│   ├── LinearStrategy           # Implements u = -Kx
│   ├── LinearStrategyTrajectory # Specialization with LinearStrategy
│   └── QuadraticCost            # Implements integral of xᵀQx + uᵀRu
│
├── SystemTrajectory             # Container for time-state-control trajectories
│
├── Solvers
│   ├── solver                   # iterative equilibrium solvers
│   └── groebner                 # symbolic equilibrium computation for small systems
│
└── Utilities
    ├── SweepRunner          # Helper for large-scale parallelized simulations
    └── Logger               # Data-Logging
```

## Support
For help and support don't hesitate to contact me:\
Felix Thömmes\
felix.thoemmes@kit.edu