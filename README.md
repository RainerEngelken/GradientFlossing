# Gradient Flossing: Improving Gradient Descent through Dynamic Control of Jacobians

This repository contains the implementation code for manuscript: <br>
__Gradient Flossing: Improving Gradient Descent through Dynamic Control of Jacobians__ <br>
## Overview
In this work we show that RNN training can be improved by pushing Lyapunov exponents of the forward pass to zero during or before training.

## Installation

#### Prerequisites
- Download [Julia](https://julialang.org/downloads/) 

#### Dependencies
- Julia (1.6)
- Flux, BackwardsLinalg

## Getting started
To install the required packages, run the following in the julia REPL after installing Julia:

```
using Pkg

for pkg in ["Flux", "BackwardsLinalg"]
    Pkg.add(pkg)
end
```

For example, to train an RNN on tracking delayed XOR task, run:
```
include("GradientFlossing_ExampleCode.jl")
# setting parameters:
N, E, Ef, Ei, Ep, Ni, B, S, T, Tp, Ti, sIC, sIn, sNet, sONS, lr, b1, b2, IC, g, gbar, I1, delay, wsS, wsM, wrS, wrM, bS, bM, nLE, task, intype, Lwnt=
80, 3001, 100, 500, 500, 2, 16, 1, 300, 55, 300, 1,1,1,1, 0.001f0, 0.9, 0.999, 1, 1.0, 0.0, 1.0,10, 1.0f0, 0.0f0, 1.0f0, 0.0f0, 0.1f0, 0.0f0,75, -1, 3, 0.0

trainRNNflossing(N, E, Ef, Ei, Ep, Ni, B, S, T, Tp, Ti, sIC, sIn, sNet, sONS, lr, b1, b2, IC, g, gbar, I1, delay, wsS, wsM, wrS, wrM, bS, bM, nLE, task, intype, Lwnt)
```

## Repository Overview

GradientFlossing_XOR.jl - generates input and target output for copy task and delayed XOR task.\
GradientFlossing_ExampleCode.jl - trains an RNN on the delayed XOR task with gradient flossing.

_GradientFlossing_ExampleCode.jl_:\
Example scripts for training networks with gradient flossing before training, with gradient flossing before and during training and without gradient flossing.
<!---

runOneStimulus.jl trains an RNN on tracking one OU-signal showing that the network becomes more tightly balanced over training epochs.\
runTwoStimuli.jl trains an RNN on two OU-signal stimulus showing that the network becomes more tightly balanced over training epochs and breaks up into two weakly-connected subnetworks.\
runTheeStimuli.jl trains an RNN on two OU-signal stimulus showing that the network becomes more tightly balanced over training epochs and breaks up into three weakly-connected subnetworks.\
![Training RNN on two signals leads to balanced subpopulations](/figures/S=2.svg?raw=true "balanced subnetworks emerge  after runTheeStimuli.jl")
-->


<!---

### Training dynamics of eigenvalues:
Here is a visualization of the recurrent weight matrix and the eigenvalues throughout across training epochs.
![Training dynamics of networks trained on multiple signals shows first tracking of global mean input](eigenvalue_movie_2D_task.gif)
-->


### Implementation details
A full specification of packages used and their versions can be found in _packages.txt_ .\
For learning rates the default ADAM parameters were used to avoid any impression of fine-tuning.\
All simulations were run on a single CPU and took on the order of minutes to a few of hours.



<!---
### figures/
Contains all figures of the main text and the supplement.
-->


<!---
### tex/
Contains the raw text of the main text and the supplement.
-->
