# Gradient Flossing: Improving Gradient Descent through Dynamic Control of Jacobians

This repository contains the implementation code for manuscript: <br>
__Gradient Flossing: Improving Gradient Descent through Dynamic Control of Jacobians__ <br>
## Overview
In this work we show that RNN training can be improved by pushing Lyapunov exponents of the forward pass to zero during or before training.

## Installation

#### Prerequisites
- Download [Julia](https://julialang.org/downloads/) 

#### Dependencies
- Julia (>= 1.5, tested on 1.6)
- Flux, PyPlot, Statistics, DelimitedFiles,RandomNumbers, ForwardDiff, BSON
- scipy

## Getting started
To install the required packages, run the following in the julia REPL after installing Julia:

```
using Pkg

for pkg in ["Flux", "PyPlot", "Statistics", "BackwardsLinalg", "RandomNumbers"]
    Pkg.add(pkg)
end
```

For example, to train an RNN on tracking delayed XOR task, run:
```
include("GradientFlossing_ExampleCode.jl")
delay=10;seed=1
trainRNNflossing(80, 301, 10, 50, 50, 2, 16, 1, 300, 55, 300, seed, seed, seed, seed, 0.001f0, 0.9, 0.999, 1, 1.0,0.0, 1.0,delay, 1.0f0, 0.0f0, 1.0f0, 0.0f0, 0.1f0, 0.0f0,75, -1, 3, 0.0)
```

## Repository Overview

### src/ 
Contains the source files.\
gf_task05spatialXORShort.jl - generates input and target output for copy task and delayed XOR task.\
GradientFlossing_ExampleCode.jl - trains a RNN on the delayed XOR task.

### example_code/
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
