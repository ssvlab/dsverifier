#!/bin/bash
export experiment_base_directory=`dirname $(readlink -f $0)`
export cbmc_directory="${experiment_base_directory}/cbmc"
export PATH="${PATH}:/${cbmc_directory}/src/cbmc:/${cbmc_directory}/src/cegis"

# Run experiments
echo -e "\nRunning Synthesis..."
benchmark_runner_directory="${experiment_base_directory}/benchmark-runner"
cd "${benchmark_runner_directory}"
chmod +x completeness-threshold-ss-benchmark-runner.sh
./completeness-threshold-ss-benchmark-runner.sh
