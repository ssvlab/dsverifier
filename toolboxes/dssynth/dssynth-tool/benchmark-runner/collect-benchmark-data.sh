#!/bin/bash

echo -e 'Benchmark ID;Success?;Runtime;Execution Date'
for benchmark_dir in ${experiment_base_directory}/benchmarks/*/; do
 ls ${benchmark_dir}*.log 1> /dev/null 2>&1 ; hasAnyLogFiles=$?
 if [ ${hasAnyLogFiles} -eq 2 ]; then continue; fi
 for benchmark_log in ${benchmark_dir}*_bound_simple.log; do
  runtime='-'
  execution_time_and_date=`date -r ${benchmark_log} +%Y-%m-%d:%H:%M`
  grep -c 'SYNTHESIS SUCCESSFUL' ${benchmark_log} >/dev/null ; success=$?
  if [ ${success} -eq 0 ]; then
   runtime=`grep '<control_synthesis_time>' ${benchmark_log} | grep -Po '(\d)*'`
  else
   benchmark_log="${benchmark_log%*_bound_simple.log}_range.log"
   if [ -f ${benchmark_log} ]; then
    grep -c 'SYNTHESIS SUCCESSFUL' ${benchmark_log} >/dev/null; success=$?
   else
    success=1
   fi
   if [ ${success} -eq 0 ]; then
    runtime=`grep '<control_synthesis_time>' ${benchmark_log} | grep -Po '(\d)*'`
   fi
  fi
  benchmark_id=$(basename $(dirname ${benchmark_log}))/$(basename ${benchmark_log})
  if [ ${success} -eq 0 ]; then success='true'; else success='false'; fi
  if [ -z "${runtime}" ]; then runtime='-'; else runtime="${runtime}s"; fi
  echo -e "${benchmark_id};${success};${runtime};${execution_time_and_date}"
 done
done
