#!/bin/bash

# compile main sbm-bp code
cmake .; make

# compile old mode-net code
pushd src/old/; g++ -O3 -c bm.cpp -o bm.o; g++ -o "./"/sbm -O3 sbm.cpp bm.o; popd
