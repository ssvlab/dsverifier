#!/bin/bash
#
# DSVerifier - Benchmark Runner
#
#               Universidade Federal do Amazonas - UFAM
# Author:       Hussama Ismail <hussamaismail@gmail.com>
#               Iury Bessa <iury.bessa@gmail.com>
#		Lennon Chaves <lennon.correach@gmail.com>
#
# ------------------------------------------------------
#
# Script to run many benchmarks using CBMC in DSVerifier
# 
# Usage Example:
# $ sh script benchmark_file.c
#
# ------------------------------------------------------

if test $# -ne 1; then
   echo "usage: $0 file_benchmark.c" >&2
   exit 1
fi

PROPERTIES=(LIMIT_CYCLE)
REALIZATIONS=(DFI DFII TDFII)
TIMEOUT="3600" # seconds
X_SIZE=20

## default parameters
# PROPERTIES=( OVERFLOW ZERO_INPUT_LIMIT_CYCLE TIMING STABILITY MINIMUM_PHASE );
# REALIZATIONS=( DFI DFII TDFII DDFI DDFII TDDFII );

BENCHMARKS_LIBRARY="$1";
BENCHMARKS=$(cat $BENCHMARKS_LIBRARY | grep 'DS_ID\|IMPLEMENTATION_COUNT')
TOTAL_DS=$(echo "$BENCHMARKS" | grep 'DS_ID' | wc -l)
OUTPUT_LOGS_DIRECTORY="./logs"
DSV_EXECUTABLE="dsverifier"
BMC_EXECUTABLE="CBMC"
BMC_PARAMETERS="-DBMC=CBMC"

# header 
INITIAL_TIMESTAMP=$(date +%s)
CPU_CORE_NUMBER=$(cat /proc/cpuinfo | grep processor | wc -l)
CPU_INFO="CPU:$(cat /proc/cpuinfo | grep "model name" | tail -n1 | cut -d ":" -f2)"
MEM_INFO="RAM: $(cat /proc/meminfo | grep "MemTotal" | cut -d ":" -f2 | cut -d " " -f8) kB"
echo "*** DSVerifier Benchmark Runner v2.0 ***"
echo ""
echo "Date of run: $(date)"
echo "System: $CPU_INFO $MEM_INFO"
echo "Source: $@ "
echo "Total of digital systems: $TOTAL_DS"
echo "";

TOTAL_EXECUTIONS=0;
TOTAL_SUCCESS=0;
TOTAL_FAILS=0;
TOTAL_TIMEOUTS=0;

AUX=2;
while [ $AUX -le $(($TOTAL_DS * 2)) ]; do

   # get the digital systems
   CURRENT=$(echo "$BENCHMARKS" | head -n $AUX | tail -n 2);
   CURRENT_DS_ID=$(echo "$CURRENT"  | grep 'DS_ID' | sed -e 's/^[ \t]*//g' |  sed -e 's/ //g' | cut -d "=" -f3);
   TOTAL_OF_IMPLEMENTATIONS=$(echo "$CURRENT"  | grep 'IMPLEMENTATION_COUNT' | cut -d " " -f2);

   # get implemenentations 
   for CURRENT_IMPLEMENTATION in $(seq 1 $TOTAL_OF_IMPLEMENTATIONS); do

       # get realizations	
       for CURRENT_REALIZATION in ${REALIZATIONS[@]}; do

           # get properties
           for CURRENT_PROPERTY in ${PROPERTIES[@]}; do

              echo "RUNNING: digital system $CURRENT_DS_ID using implementation $CURRENT_IMPLEMENTATION in $CURRENT_REALIZATION realization checking $CURRENT_PROPERTY property";

              mkdir -p $OUTPUT_LOGS_DIRECTORY/$CURRENT_PROPERTY;
              OUT_FILE="$OUTPUT_LOGS_DIRECTORY/$CURRENT_PROPERTY/result-ds$CURRENT_DS_ID-impl$CURRENT_IMPLEMENTATION-$CURRENT_REALIZATION.out";   

              # do the verification
              INITIAL_EXECUTION_TIMESTAMP=$(date +%s)
              timeout $TIMEOUT bash -c "{ time $DSV_EXECUTABLE $BENCHMARKS_LIBRARY $BMC_PARAMETERS -DDS_ID=$CURRENT_DS_ID -DIMPLEMENTATION_ID=$CURRENT_IMPLEMENTATION --realization $CURRENT_REALIZATION --property $CURRENT_PROPERTY --x-size $X_SIZE --show-ce-data> $OUT_FILE; } 2>> $OUT_FILE ";
              FINAL_EXECUTION_TIMESTAMP=$(date +%s)

              # analyse the result 
              OUT=$(cat $OUT_FILE);	
              TIME=$((FINAL_EXECUTION_TIMESTAMP - INITIAL_EXECUTION_TIMESTAMP));
              VERIFICATION_SUCCESSFUL=$(echo "$OUT" | grep "SUCCESSFUL" | wc -l); 
              VERIFICATION_FAILED=$(echo "$OUT" | grep "FAILED" | wc -l);
              VERIFICATION_TIMEOUT=$(echo "$OUT" | grep "Timed out" | wc -l);

              # check if manual timeout for cbmc
              if [ $VERIFICATION_SUCCESSFUL -eq 0 ] && [ $VERIFICATION_FAILED -eq 0 ] && [ $VERIFICATION_TIMEOUT -eq 0 ]; then
                  TIME_PARAMETERS=$(cat $OUT_FILE | grep -w "real\|user\|sys" | wc -l)
                  if [ $TIME_PARAMETERS -lt 3 ]; then
                     VERIFICATION_TIMEOUT=1;
                  fi               
              fi

              TOTAL_EXECUTIONS=$((TOTAL_EXECUTIONS + 1));
              if [ $VERIFICATION_SUCCESSFUL -eq 1 ]; then
            	  TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1));
                  echo "$(echo -e "\033[0;32msuccess\033[0m" | cut -d " " -f2) in "$TIME"s ($OUT_FILE)";       
              elif [ $VERIFICATION_FAILED -eq 1 ]; then
            	  TOTAL_FAILS=$((TOTAL_FAILS + 1));
                  echo "$(echo -e "\033[0;31mfail\033[0m" | cut -d " " -f2) in "$TIME"s ($OUT_FILE)";
              elif [ $VERIFICATION_TIMEOUT -eq 1 ]; then
            	  TOTAL_TIMEOUTS=$((TOTAL_TIMEOUTS + 1));
                  echo "$(echo -e "\033[1;35mtimeout\033[0m" | cut -d " " -f2) in "$TIME"s ($OUT_FILE)";
              else
                  echo "$(echo -e "\033[0;33munknown\033[0m" | cut -d " " -f2) ($OUT_FILE)"; 
              fi

           done
       done;
   done;
   
   AUX=$((AUX+2));
done 

# show report
echo "";
echo "*** RESULTS *** ";
echo "Total of verifications: $TOTAL_EXECUTIONS";
echo "Total of success: $TOTAL_SUCCESS";
echo "Total of fails: $TOTAL_FAILS";
echo "Total of timeouts: $TOTAL_TIMEOUTS";
