#!/bin/bash
#
# DSSynth - Script Results Extractor
#
# Author: Lennon Chaves <lennon.correach@gmail.com>
#
# ------------------------------------------------------
#
# Script to extract data from log file considering the PATH
# 
# Usage Example:
# $ sh script $PATH
#
# ------------------------------------------------------

OUTPUT_FILE=$1;

if [ -z $OUTPUT_FILE ]; then
	echo "Inform a directory for extract the results (use: $0 \$PATH)";
	exit 1;
fi

NUMERATOR=$(cat $OUTPUT_FILE | grep "    .num={" | cut -d "=" -f2 | sed 's/{//' | sed 's/}//' | sed 's/.num_uncertainty//' | tr ',' ' ');
DENOMINATOR=$(cat $OUTPUT_FILE | grep "  controller={ .den={ " | cut -d "=" -f3 | sed 's/{//' | sed 's/}//' | sed 's/.den_uncertainty//' | tr ',' ' ');

echo $NUMERATOR "\n" $DENOMINATOR > controller.out;
