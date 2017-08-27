#!/bin/sh
# script to run all test cases
#
# run chmod a+x run-all.sh
# ./run-all.sh
#
# author: Lennon Chaves
# August, 2017
# Manaus, Amazonas
#

chmod a+x boolector/run-all.sh
chmod a+x z3/run-all.sh
path=$PWD

echo "RUNNING ESBMC TESTS - Boolector";
cd $path
cd $path/boolector
./run-all.sh

echo "RUNNING ESBMC TESTS - Z3";
cd $path
cd $path/z3
./run-all.sh
