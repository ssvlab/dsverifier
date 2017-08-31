#!/bin/bash
status_output_file='output.txt'
synthesis_file='FWL_LTI.c'
width_check_file='width_LTI.c'
script_base_directory=`pwd`
spec_header_file='spec.h'
cbmc_log_file='cbmc-tmp.log'
solution_file='solution.log'

function setup_benchmark_directory {
 mkdir -p "$1" 2>/dev/null
 cp ${script_base_directory}/AACegar/* ${working_directory}/ 2>/dev/null
 chmod +x ${working_directory}/axelerator
}

function echo_log {
 echo $1 | tee -a ${log_file}
}

function echo_success_message {
 start_time=$1
 end_time=$2
 runtime=$((end_time-start_time))
 echo_log 'SYNTHESIS SUCCESSFUL'
 echo_log "<control_synthesis_time>${runtime}</control_synthesis_time>"
}

function extract_spec_scalar {
 echo $1 | sed "s/.*$2 *= *\([^;]*\);.*/\1/"
}

function extract_spec_matrix {
 echo $1 | sed "s/.*$2 *= *\[\([^]]*\)\].*/\1/"
}

function extract_int_bits {
 echo $1 | sed "s/.*implementation *< *\([^,]*\),.*/\1/"
}

function extract_frac_bits {
 echo $1 | sed "s/.*implementation *< *[^,]*,\([^> ]*\) *>.*/\1/"
}

function extract_input_lower_bound {
 echo $1 | sed "s/.*inputs *= *\[ *\([^,]*\) *,.*/\1/"
}

function extract_input_upper_bound {
 echo $1 | sed "s/.*inputs *= *\[ *[^,]*,\([^]]*\).*/\1/"
}

function write_spec_header {
 header_file="${working_directory}/${spec_header_file}"
 echo "#define INT_BITS ${impl_int_bits}" >>${header_file}
 echo "#define FRAC_BITS ${impl_frac_bits}" >>${header_file}
 echo '#include "control_types.h"' >>${header_file}
 echo '' >>${header_file}
 echo "#define NSTATES ${num_states}" >${header_file}
 echo "#define NINPUTS ${num_inputs}" >>${header_file}
 echo "#define NOUTPUTS ${num_outputs}" >>${header_file}
 echo "#define INPUT_LOWERBOUND (__CPROVER_EIGEN_fixedbvt)$3" >>${header_file}
 echo "#define INPUT_UPPERBOUND (__CPROVER_EIGEN_fixedbvt)$4" >>${header_file}
 A_value=$(echo $1 | sed -r 's/;/ }, { /g')
 echo "const __CPROVER_EIGEN_fixedbvt _controller_A[NSTATES][NSTATES] = { { ${A_value} } };" >>${header_file}
 B_value=$(echo $2 | sed -r 's/;/, /g')
 echo "const __CPROVER_EIGEN_fixedbvt _controller_B[NSTATES] = { ${B_value} };" >>${header_file}
}

function get_current_cpu_millis {
 cat ${time_tmp_file} >>${log_file}
 formula=$(sed -r 's/([0-9]+(\.[0-9]+)?)m([0-9]+)\.0*([0-9]+)?s/60000*\1+1000*\3+\4/g' ${time_tmp_file} | tr '\n' ' ' | sed -r 's/ /+/g' | sed -r 's/(.*)[+]$/\1/' | sed -r 's/(.*)[+]$/\1/' | sed -r 's/[+]+/+/')
 echo $((${formula}))
}

if [ -z "$1" ]; then
 benchmark_dirs=("${experiment_base_directory}/benchmarks/state-space/pendulum_ss/") #ok
else
 #benchmark_dirs=("$1")
 working_directory_base_suffix="$1"
 #dkr10
 if [ "$1" == "dkr10" ]; then
  benchmark_dirs=("${experiment_base_directory}/benchmarks/state-space/cruise_ss/" "${experiment_base_directory}/benchmarks/state-space/dcmotor_ss/" "${experiment_base_directory}/benchmarks/state-space/helicopter_ss/")
 fi

 #dkr11
 if [ "$1" == "dkr11" ]; then
  #benchmark_dirs=("${experiment_base_directory}/benchmarks/state-space/ballmaglev_ss/" "${experiment_base_directory}/benchmarks/state-space/magsuspension_ss/" "${experiment_base_directory}/benchmarks/state-space/invpendulum_pendang_ss/")
  benchmark_dirs=("${experiment_base_directory}/benchmarks/state-space/magsuspension_ss/" "${experiment_base_directory}/benchmarks/state-space/invpendulum_pendang_ss/")
 fi

 #dkr12
 if [ "$1" == "dkr12" ]; then
  #benchmark_dirs=("${experiment_base_directory}/benchmarks/state-space/magneticpointer_ss/" "${experiment_base_directory}/benchmarks/state-space/invpendulum_cartpos_ss/" "${experiment_base_directory}/benchmarks/state-space/suspension_ss/")
  benchmark_dirs=("${experiment_base_directory}/benchmarks/state-space/suspension_ss/")
 fi

 #dkr13
 if [ "$1" == "dkr13" ]; then
  benchmark_dirs=("${experiment_base_directory}/benchmarks/state-space/pendulum_ss/" "${experiment_base_directory}/benchmarks/state-space/uscgtampa_ss/")
  #benchmark_dirs=("${experiment_base_directory}/benchmarks/state-space/satellite_ss/" "${experiment_base_directory}/benchmarks/state-space/tapedriver_ss/" "${experiment_base_directory}/benchmarks/state-space/pendulum_ss/" "${experiment_base_directory}/benchmarks/state-space/uscgtampa_ss/")
 fi
fi

working_directory_base="/tmp/control_synthesis-ss-${working_directory_base_suffix}"
mkdir -p ${working_directory_base} 2>/dev/null

for benchmark_dir in ${benchmark_dirs[@]}; do
 for benchmark in ${benchmark_dir}*.ss; do
  log_file="${benchmark%.ss}_accelerator-ss.log"
  time_tmp_file=/tmp/times${working_directory_base_suffix}.log
  times >${time_tmp_file}; start_time=$(get_current_cpu_millis)
  truncate -s 0 ${log_file}
  echo_log ${benchmark}
  echo_log 'accelerator-ss'

  spec_content=`cat ${benchmark}`
  impl_int_bits=$(extract_int_bits "${spec_content}")
  impl_frac_bits=$(extract_frac_bits "${spec_content}")
  controller_fixedbv_type_width=$((impl_int_bits+impl_frac_bits))
  width_offset=$((controller_fixedbv_type_width%8))
  [ ${width_offset} -ne 0 ] && impl_int_bits=$((impl_int_bits+8-width_offset))
  num_states=$(extract_spec_scalar "${spec_content}" 'states')
  num_inputs=$(extract_spec_scalar "${spec_content}" 'inputs')
  num_outputs=$(extract_spec_scalar "${spec_content}" 'outputs')
  A=$(extract_spec_matrix "${spec_content}" 'A')
  B=$(extract_spec_matrix "${spec_content}" 'B')
  input_lower_bound=$(extract_input_lower_bound "${spec_content}")
  input_upper_bound=$(extract_input_upper_bound "${spec_content}")
  options="-u -p -ii -mpi 256 -synth CEGIS -params \"p=${num_states},q=${num_inputs},f=1,m=256:$((impl_int_bits+impl_frac_bits)):${impl_int_bits}\" -dynamics \"[${A}]\" -init \"[cube<.5]\" -isense \"[${B}]\" -inputs \"[1>${input_lower_bound};1<${input_upper_bound}]\" -sguard \"[cube<1]\""

  working_directory="${working_directory_base}/accelerator-ss"
  setup_benchmark_directory ${working_directory}
  cd ${working_directory}
  #write_spec_header "$A" "$B" "$input_lower_bound" "$input_upper_bound"

  max_length=64
  #integer_width=8
  integer_width=$((2*impl_int_bits))
  radix_width=$((2*impl_frac_bits))
  #radix_width=$((impl_int_bits+impl_frac_bits))
  min_size_offset=$(((integer_width+radix_width)%8))
  [ ${min_size_offset} -ne 0 ] && integer_width=$((integer_width+8-min_size_offset))
  timeout_time=3600
  kill_time=3780
  echo_log "./axelerator $options -control \"[0]\""
  eval "./axelerator $options -control \"[0]\""
  while [ $((integer_width+radix_width)) -le ${max_length} ]; do
   echo_log "int: ${integer_width}, radix: ${radix_width}"
   solution_found=false
   synthesis_in_progress=true
   while [ "${synthesis_in_progress}" = true ]; do
    echo_log "cbmc -D __CPROVER -D _FIXEDBV -D _CONTROL_FLOAT_WIDTH=$((integer_width+radix_width)) -D _CONTORL_RADIX_WIDTH=${radix_width} \"${working_directory}/${synthesis_file}\" --stop-on-fail >${working_directory}/${cbmc_log_file}"
    timeout --preserve-status --kill-after=${kill_time} ${timeout_time} cbmc -D __CPROVER -D _FIXEDBV -D _CONTROL_FLOAT_WIDTH=$((integer_width+radix_width)) -D _CONTORL_RADIX_WIDTH=${radix_width} "${working_directory}/${synthesis_file}" --stop-on-fail >${working_directory}/${cbmc_log_file} 2>&1
    if [ $? -eq 10 ]; then
     controller=$(grep 'controller_cbmc=' ${working_directory}/${cbmc_log_file} | sed 's/.*controller_cbmc={ *\([^}]*\) *}.*/\1/')
     echo_log "./axelerator $options -control \"[${controller}]\""
     eval "./axelerator $options -control \"[${controller}]\""
     if grep --quiet 'SUCCESS' "${working_directory}/${status_output_file}"; then
      times >${time_tmp_file}; end_time=$(get_current_cpu_millis)
      echo_success_message ${start_time} ${end_time}
      echo_log "<controller>${controller}</controller>"
      echo_log "msg = '$(basename ${benchmark})'"
      echo "msg = '$(basename ${benchmark})'" >>${solution_file}
      matlab_controller="[$(echo ${controller} | sed -r 's/(-?[0-9]+(\.[0-9e-]+)?)/fxp(\1)/g')]"
      echo_log "K = ${matlab_controller};"
      echo "K = ${matlab_controller};" >>${solution_file}
      echo_log "A = [ ${A} ];"
      echo "A = [ ${A} ];" >>${solution_file}
      echo_log "B = [ ${B} ];"
      echo "B = [ ${B} ];" >>${solution_file}
      echo_log "INPUT = [${input_lower_bound},${input_upper_bound}];"
      echo "INPUT = [${input_lower_bound},${input_upper_bound}];" >>${solution_file}
      echo "run fixedpointcheck.m" >>${solution_file}
      echo "" >>${solution_file}
      
      solution_found=true
      synthesis_in_progress=false
     else
      if grep --quiet 'DIVERGENT' "${working_directory}/system.h"; then
       solution_found=false;
       synthesis_in_progress=false;
      else
       if grep --quiet 'NO_FEEDBACK' "${working_directory}/system.h"; then
        solution_found=false;
        synthesis_in_progress=false
       else
        echo_log "<controller>${controller}</controller>"      
        echo_log 'Refining abstraction...'
       fi       
      fi
     fi
    else
     synthesis_in_progress=false
    fi
   done
   if [ "${solution_found}" = true ]; then
    break
   fi
   integer_width=$((integer_width+4))
   radix_width=$((radix_width+4))
  done
  # All files are the same benchmark with increased sample frequency. Exit after first.
  if [ "${solution_found}" = true ]; then
    break
  fi
 done
done
