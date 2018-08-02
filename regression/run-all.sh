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

chmod a+x cbmc/run-all.sh
chmod a+x esbmc/run-all.sh

echo "RUNNING CBMC TESTS";
echo "";
path=$PWD
cd $path/cbmc
./run-all.sh

echo "RUNNING ESBMC TESTS";
echo "";
cd $path
cd $path/esbmc
./run-all.sh
