#!/bin/bash
export experiment_base_directory=`dirname $(readlink -f $0)`

# Compile CBMC/CEGIS
cbmc_directory="${experiment_base_directory}/cbmc"
cd "${cbmc_directory}/src"
export PATH="${PATH}:/${cbmc_directory}/src/cbmc:/${cbmc_directory}/src/cegis"

# Run experiments
echo -e "\nRunning Synthesis..."
benchmark_runner_directory="${experiment_base_directory}/benchmark-runner"
cd "${benchmark_runner_directory}"
./benchmark-runner.sh simple
#./benchmark-runner.sh range

echo -e "\nBenchmark results as CSV:"
# Collect and display experiments results
./collect-benchmark-data.sh
