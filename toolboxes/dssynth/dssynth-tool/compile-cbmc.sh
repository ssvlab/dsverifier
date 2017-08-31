#!/bin/bash
export experiment_base_directory=`dirname $(readlink -f $0)`

# Compile CBMC/CEGIS
cbmc_directory="${experiment_base_directory}/cbmc"
cd "${cbmc_directory}/src"
#make minisat2-download
make -j 4 all
make -j 4 cegis.dir

# Compile axelerator
axelerator_directory="${experiment_base_directory}/benchmark-runner/AACegar/"
cd "${axelerator_directory}"
make -j 4
