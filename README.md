# sbm-bp  [![Build Status](https://travis-ci.org/junipertcy/sbm-bp.svg?branch=master)](https://travis-ci.org/junipertcy/sbm-bp)

**sbm-bp** implements the belief propagation algorithm for the inference of the (degree-corrected) stochastic block model. 
This program is largely an object-oriented re-implemetation of the ["MODE-NET" code](http://mode_net.krzakala.org/) by Aurelien Decelle, Florent Krzakala and Pan Zhang. 

This program is tested with a few benchmark networks, including synthetic networks and the karate club network. 
It generates nearly identical outputs as the original ones. For completeness, the original code is also included in this repository, under `src/old`.

Documentation will be updated soon.

## Table of content

1. [Usage](#usage)
2. [Compilation](#compilation)
3. [Companion articles](#refereces)


## Usage

### Compilation

This code requires [compilers](http://en.cppreference.com/w/cpp/compiler_support) that support C++14 features. 
It also depends on `boost::program_options` and `cmake`.

Compilation:
```commandline
cmake .
make
```

The binaries are built in `bin/`.

### Inference
```commandline
bin/bp -l dataset/N_1000-Q_2-method_cab_ec-eps_0.1-c_3.0.edgelist -n 500 500 --pa 0.5 0.5 --cab 3.63 2.36 3.63 -t 1000 -i 0 --deg_corr_flag 0 -m infer
```

### Learning
```commandline
bin/bp -l dataset/N_1000-Q_2-method_cab_ec-eps_0.1-c_3.0.edgelist -n 500 500 --pa 0.5 0.5 --cab 3.63 2.36 3.63 -t 1000 -i 0 --deg_corr_flag 0 -m learn
```

## References

Please find the references from the [original project page](http://mode_net.krzakala.org/).
