# gamecore — Core Library for Dynamic and Differential Games

## Description
A Python-based library for simulating Differential and Dynamic Games.
This package builds a shared core library of the "Cooperative Systems" research group at the Institute of Control Systems (IRS), Karlsruhe Institute of Technology (KIT), Germany.

## Installation

`gamecore` is distributed via a **public mirror repository**.
Install the newest version, i.e., the current main, with
```bash
pip install "gamecore @ git+https://github.com/felixthoe/gamecore.git"
```
Or, install a specific released version with, e.g.:
```bash
pip install "gamecore @ git+https://github.com/felixthoe/gamecore.git@v0.1.4"
```
We recommend to always use a tagged version for reproducibility.

## Usage
### Getting Started
You can get to know this codebase with the tutorial notebooks in `notebooks/tutorials/`.

To use the package in your code, simply:
```python
import gamecore
```

## Overall Project Structure

The `gamecore` package provides the following conceptual building blocks:

```text
gamecore
├── Game abstraction
│   ├── BaseGame                    # Holds BaseSystem and BasePlayer[]
│   ├── BaseSystem                  # Defines the ODEs / system dynamics
│   ├── BasePlayer                  # Holds a strategy and a cost function
│   ├── BaseStrategy                # Abstract interface for control laws
│   ├── BaseStrategyTrajectory      # Tracking and reconstruction of learned BaseStrategies and costs
│   └── BaseCost                    # Abstract interface for cost functions
│
├── Linear-Quadratic specialization
│   ├── LQGame                      # Specialization with LinearSystem and LQPlayer[]
│   ├── LinearSystem                # Linear version of BaseSystem
│   ├── LQPlayer                    # Player with LinearStrategy and QuadraticCost
│   ├── LinearStrategy              # Implements u = -Kx
│   ├── LinearStrategyTrajectory    # Specialization with LinearStrategy
│   └── QuadraticCost               # Implements integral of xᵀQx + uᵀRu
│
├── SystemTrajectory                # Container for time-state-control trajectories
│
├── Solvers
│   ├── solver                      # iterative equilibrium solvers
│   └── groebner                    # symbolic equilibrium computation for small systems
│
└── Utilities
    ├── SweepRunner                 # Helper for large-scale parallelized simulations
    └── Logger                      # Data-Logging
```

## Repository Structure and Mirroring

This project is developed in a **private internal repository** on KIT's Gitlab
within the Cooperative Systems research group.

A (read-only) **public mirror** is maintained on Github for released versions to allow
easy installation, reproducibility, and external use.


## Development Workflow (Internal Development)

This repository is intended to be a **shared and stable core library**.
To ensure consistency, reproducibility, and a clean API, changes to the codebase
should follow the workflow described below.

### 1. Cloning the Repository

Development takes place in the internal repository. 
To work on `gamecore`, first clone the internal repository locally (requires access and confidentials):

```bash
git clone git@gitlab.kit.edu/kit/irs/rus/ks/gamecore.git
cd gamecore
```
The public repository is a mirror and should not be used for development.

### 2. Creating a Feature/Bugfix Branch
All changes should be made on a separate branch.
Do not commit directly to `main`.
```bash
git checkout -b feature/<short-description>
```
E.g., feature/new-sweep-runner, bugfix/fix-linear-system-dimensions, improvement/efficient_cost_generation, ...

### 3. Making Changes 
Keep changes focused and minimal.
When finished, ensure that all existing tests pass and add tests where appropriate.
Run tests locally before pushing:
```bash
pytest
```

### 4. Pushing and Creating a Merge Request
Push your branch to GitLab:
```bash
git push origin feature/<short-description>
```
Then open a Merge Request (Pull Request) against `main` on Gitlab.

### 5. Merging and Versioning
After a Merge Request is approved and merged into `main`,
the version number should be updated and a new tag created.
We follow a semantic versioning scheme:
- PATCH (0.1.x) — bug fixes, minor changes, no API changes
- MINOR (0.x.0) — new features, backwards compatible
- MAJOR (x.0.0) — breaking API changes

Create a new version based on the udpated main with:
```bash
git checkout main
git pull
git tag vx.x.x
git push origin vx.x.x
```
The mirror repository on Github gets updated automatically.

## Support
For help and support don't hesitate to contact me:\
Felix Thömmes\
felix.thoemmes@kit.edu