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

CONTROLLER=$(cat $OUTPUT_FILE | grep "K = " | cut -d "=" -f2 | sed 's/(//g' | sed 's/)//g' | sed -r 's/fxp//g' );


echo $CONTROLLER > controller.out;
