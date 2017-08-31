#!/bin/bash
synthesis_file='safety_stability.c'
script_base_directory=`pwd`
spec_header_file='benchmark.h'
source_template_directory=${script_base_directory}/universalrunner
solution_file='solution.log'

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

function setup_benchmark_directory {
 mkdir -p "$1" 2>/dev/null
 #rm "$1"/* 2>/dev/null
 cp ${source_template_directory}/${synthesis_file} ${working_directory}/
 cp ${source_template_directory}/control_types.h ${working_directory}/
 cp ${source_template_directory}/operators.h ${working_directory}/
 cp ${source_template_directory}/intervals.h ${working_directory}/
 cp ${source_template_directory}/mpreal.h ${working_directory}/
 cp ${source_template_directory}/int_2Inverse.h ${working_directory}/
 cp ${source_template_directory}/int_3Inverse.h ${working_directory}/
 cp ${source_template_directory}/discrete_step_k_completeness_check.cpp ${working_directory}/
}

function write_spec_header {
 header_file="${working_directory}/${spec_header_file}"
 echo "#define INT_BITS ${impl_int_bits}" >${header_file}
 echo "#define FRAC_BITS ${impl_frac_bits}" >>${header_file}
 echo "#define NSTATES ${num_states}" >>${header_file}
 echo '#include "control_types.h"' >>${header_file}
 echo '' >>${header_file}

 echo "#define NINPUTS ${num_inputs}" >>${header_file}
 echo "#define NOUTPUTS ${num_outputs}" >>${header_file}
 echo '#ifdef INTERVAL' >>${header_file}
 echo "#define INPUT_LOWERBOUND (__plant_precisiont)${input_lower_bound}" >>${header_file}
 echo "#define INPUT_UPPERBOUND (__plant_precisiont)${input_upper_bound}" >>${header_file}
 echo '#else' >>${header_file}
 echo "#define INPUT_LOWERBOUND (__plant_typet)${input_lower_bound}" >>${header_file}
 echo "#define INPUT_UPPERBOUND (__plant_typet)${input_upper_bound}" >>${header_file}
 echo '#endif' >>${header_file}
 A_value=$(echo ${A} | sed -r 's/;/ }, { /g')
 A_value=$(echo ${A_value} | sed -r 's/([-0-9]+(\.[0-9e-]+)?)/interval(\1)/g')
 echo "const __plant_typet _controller_A[NSTATES][NSTATES] = { { ${A_value} } };" >>${header_file}
 B_value=$(echo ${B} | sed -r 's/;/, /g')
 B_value=$(echo ${B_value} | sed -r 's/([-0-9]+(\.[0-9e-]+)?)/interval(\1)/g')
 echo "const __plant_typet _controller_B[NSTATES] = { ${B_value} };" >>${header_file}

 #g++ -I . -I /usr/include/eigen3/ discrete_step_k_completeness_check.cpp -o discrete_step_k_completeness_check -lmpfr
 #chmod +x discrete_step_k_completeness_check
 gcc -D INTERVAL safety_stability.c -o precision_check
 chmod +x precision_check
}

function get_current_cpu_millis {
 cat ${time_tmp_file} >>${log_file}
 formula=$(sed -r 's/([0-9]+(\.[0-9]+)?)m([0-9]+)\.0*([0-9]+)?s/60000*\1+1000*\3+\4/g' ${time_tmp_file} | tr '\n' ' ' | sed -r 's/ /+/g' | sed -r 's/(.*)[+]$/\1/' | sed -r 's/(.*)[+]$/\1/' | sed -r 's/[+]+/+/')
 echo $((${formula}))
}

if [ -z "$1" ]; then
 benchmark_dirs=("${experiment_base_directory}/benchmarks/system/") #ok
fi

working_directory_base="/tmp/control_synthesis-ss-${working_directory_base_suffix}"
mkdir -p ${working_directory_base} 2>/dev/null

for benchmark_dir in ${benchmark_dirs[@]}; do
 for benchmark in ${benchmark_dir}*.ss; do
  log_file="${benchmark%.ss}_completeness-threshold-ss.log"
  time_tmp_file=/tmp/times${working_directory_base_suffix}.log
  times >${time_tmp_file}; start_time=$(get_current_cpu_millis)
  truncate -s 0 ${log_file}
  echo_log ${benchmark}
  echo_log 'completeness-threshold-ss'

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

  working_directory="${working_directory_base}/completeness-threshold-ss"
  setup_benchmark_directory ${working_directory}
  cd ${working_directory}
  write_spec_header
  cbmc_log_file="${working_directory}/cbmc.log"

  max_length=64
  #integer_width=8
  integer_width=${impl_int_bits}
  radix_width=$((impl_int_bits+impl_frac_bits))
  #radix_width=${impl_frac_bits}
  min_size_offset=$(((integer_width+radix_width)%8))
  [ ${min_size_offset} -ne 0 ] && integer_width=$((integer_width+8-min_size_offset))
  k_sizes=(10 20 30 50 75 100 200)
  k_size_index=0
  timeout_time=3600
  kill_time=3780
  while [ $((integer_width+radix_width)) -le ${max_length} ] && [ ${k_size_index} -lt ${#k_sizes[@]} ]; do
   k_size=${k_sizes[${k_size_index}]}
   echo_log "int: ${integer_width}, radix: ${radix_width}"
   solution_found=false
   echo_log "cegis --round-to-minus-inf --cegis-control --cegis-statistics --cegis-max-size 1 --cegis-show-iterations -D CPROVER -D _CONTROL_FLOAT_WIDTH=$((integer_width+radix_width)) -D _CONTORL_RADIX_WIDTH=${radix_width} -D NUMBERLOOPS=${k_size} ${synthesis_file}"
   timeout --preserve-status --kill-after=${kill_time} ${timeout_time} cegis --round-to-minus-inf --cegis-control --cegis-statistics --cegis-max-size 1 --cegis-show-iterations -D CPROVER -D _CONTROL_FLOAT_WIDTH=$((integer_width+radix_width)) -D _CONTORL_RADIX_WIDTH=${radix_width} -D NUMBERLOOPS=${k_size} ${synthesis_file} 2>>${log_file} 1>${cbmc_log_file}
   cbmc_result=$?
   cat ${cbmc_log_file} >>${log_file}
   controller_items=$(grep '<item>' ${cbmc_log_file} | tail -n ${num_states})
   controller_params=$(echo "${controller_items}" | sed -r 's/<\/item> <item>/ /g' | sed -r 's/<item>//g' | sed -r 's/<\/item>//g' | tr '\n' ' ')
   if [ ${cbmc_result} -eq 0 ]; then
    #echo_log "./discrete_step_k_completeness_check ${k_size} ${controller_params}"
    #eval "./discrete_step_k_completeness_check ${k_size} ${controller_params}"
    k_check_result=$?
    echo_log "k_check_result: ${k_check_result}"
    if [ ${k_check_result} -eq 2 ]; then
     k_size_index=$((k_size_index+1))
    elif [ ${k_check_result} -eq 1 ]; then
     echo_log 'K check error occurred'
     break
    else
     #controller_intervals=$(echo "${controller_params}" | sed -r 's/([-0-9]+(\.[0-9]*)?)/interval(\1),/g' | sed 's/\(.*\),/\1/')
     #echo "controller_intervals: ${controller_intervals}"
     #gcc -D INTERVAL -D _CONTROL_FLOAT_WIDTH=$((integer_width+radix_width)) -D _CONTORL_RADIX_WIDTH=${radix_width} -D NUMBERLOOPS=${k_size} -D _CONTROLLER_INTERVALS="${controller_intervals}" safety_stability.c -o precision_check
     echo_log "./precision_check ${k_size} ${controller_params}"
     ./precision_check ${k_size} ${controller_params}
     if [ $? -eq 0 ]; then
      times >${time_tmp_file}; end_time=$(get_current_cpu_millis)
      echo_success_message ${start_time} ${end_time}
      echo_log "<controller>${controller_params}</controller>"
      echo_log "msg = '$(basename ${benchmark})'"
      echo "msg = '$(basename ${benchmark})'" >>${solution_file}
      matlab_controller="[$(echo ${controller_items} | sed -r 's/(-?[0-9]+(\.[0-9e-]+)?)/fxp(\1)/g' | sed -r 's/<item>//g' | sed -r 's/<\/item>//g' | sed -r 's/\)/\), /g')]"
      matlab_controller="$(echo ${matlab_controller} | sed -r 's/, \]/ \]/g')"
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
      break
     else
      integer_width=$((integer_width+4))
      radix_width=$((radix_width+4))
     fi
    fi
   else
    integer_width=$((integer_width+4))
    radix_width=$((radix_width+4))
   fi
  done
  # All files are the same benchmark with increased sample frequency. Exit after first.
  if [ "${solution_found}" = true ]; then
    break
  fi
 done
done
