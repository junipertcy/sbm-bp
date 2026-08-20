# sbm-bp

**sbm-bp** implements the belief propagation algorithm for the inference of the (degree-corrected) stochastic block model.
This program is largely an object-oriented re-implementation of the "MODE-NET" code by Aurelien Decelle, Florent Krzakala, Lenka Zdeborová, and Pan Zhang.
The original project page (`mode_net.krzakala.org`) is no longer online, so for completeness and attribution, the original code is preserved in this repository under [`src/old`](src/old).

This program is tested with a few benchmark networks, including synthetic networks and the karate club network.
It generates nearly identical outputs as the original code.

## Table of contents

1. [Compilation](#compilation)
2. [Usage](#usage)
3. [References](#references)

## Compilation

This code requires a [compiler](http://en.cppreference.com/w/cpp/compiler_support) that supports C++14 features.
It also depends on `boost::program_options` and `cmake` (3.5 or later).

```commandline
cmake .
make
```

The binaries are built in `bin/`.

## Usage

Run `bin/bp -h` for the full list of options. The main flags are `-l` (path to the edgelist),
`-n` (block sizes), `--pa` (group priors), `--cab` (affinity matrix), `-t` (maximum time for BP to converge),
and `-m` (mode: `infer` or `learn`).

### Inference

Compute the marginal group memberships with the model parameters held fixed:

```commandline
bin/bp -l dataset/N_1000-Q_2-method_cab_ec-eps_0.1-c_3.0.edgelist -n 500 500 --pa 0.5 0.5 --cab 3.63 2.36 3.63 -t 1000 -i 0 --deg_corr_flag 0 -m infer
```

### Learning

Learn the model parameters by expectation-maximization, with belief propagation as the E-step:

```commandline
bin/bp -l dataset/N_1000-Q_2-method_cab_ec-eps_0.1-c_3.0.edgelist -n 500 500 --pa 0.5 0.5 --cab 3.63 2.36 3.63 -t 1000 -i 0 --deg_corr_flag 0 -m learn
```

## References

- A. Decelle, F. Krzakala, C. Moore, and L. Zdeborová, "Inference and phase transitions in the detection of modules in sparse networks," [Phys. Rev. Lett. 107, 065701 (2011)](https://doi.org/10.1103/PhysRevLett.107.065701).
- A. Decelle, F. Krzakala, C. Moore, and L. Zdeborová, "Asymptotic analysis of the stochastic block model for modular networks and its algorithmic applications," [Phys. Rev. E 84, 066106 (2011)](https://doi.org/10.1103/PhysRevE.84.066106).
- B. Karrer and M. E. J. Newman, "Stochastic blockmodels and community structure in networks," [Phys. Rev. E 83, 016107 (2011)](https://doi.org/10.1103/PhysRevE.83.016107).
- X. Yan et al., "Model selection for degree-corrected block models," [J. Stat. Mech. (2014) P05007](https://doi.org/10.1088/1742-5468/2014/05/P05007).

## See also

- [bipartiteSBM](https://github.com/junipertcy/bipartiteSBM) — Bayesian community detection for bipartite networks (model selection and MCMC sampling).

**sbm-bp** is released under the GNU General Public License v3.0.
