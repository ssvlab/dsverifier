#!/bin/bash
#
# DSVerifier - Script Results Extractor
#
#               Universidade Federal do Amazonas - UFAM
# Author:       Lennon Chaves <lennon.correach@gmail.com>
#
# ------------------------------------------------------
#
# Script to extract data from dsverifer considering the PATH
# 
# Usage Example:
# $ sh script $PATH
#
# ------------------------------------------------------


if [ -z "$1" ]; then
	echo "Inform a directory for extract the results (use: $0 \$PATH)";
	exit 1;
fi

PREVIOUS_FOLDER=$(pwd);
DESIRED_FOLDER=$1;

if [ ! -d "$DESIRED_FOLDER" ]; then
	"The directory $DESIRED_FOLDER doesn't exists";
	exit 1;
fi

cd $DESIRED_FOLDER;

FILES=$(ls -tr $(pwd) | grep out);
FOLDER=${PWD##*/}
RESULT="*.out";
EXTRACTION="\n";
N_CE=0;

for file_desired in $RESULT; do
	
OUTPUT_FILE=$file_desired;
OUT=$(cat $OUTPUT_FILE);

VERIFICATION_FAILED=$(echo "$OUT" | grep "FAILED" | wc -l);


if [ $VERIFICATION_FAILED -eq 1 ]; then

REALIZATION=$(cat $OUTPUT_FILE | grep " Realization \=" | cut -d "=" -f2 );
IMPLEMENTATION=$(cat $OUTPUT_FILE | grep " Implementation \=" | cut -d "=" -f2 | sed 's/<//' | sed 's/>//' | tr ',' ' ');
NUMERATOR=$(cat $OUTPUT_FILE | grep " Numerator  \=" | cut -d "=" -f2 | sed 's/}//' | sed 's/{//');
DENOMINATOR=$(cat $OUTPUT_FILE | grep " Denominator  \=" | cut -d "=" -f2 | sed 's/}//' | sed 's/{//');
NSIZE=$(cat $OUTPUT_FILE | grep " Numerator Size \=" | cut -d "=" -f2 );
DSIZE=$(cat $OUTPUT_FILE | grep " Denominator Size \=" | cut -d "=" -f2 );
DELTA=$(cat $OUTPUT_FILE | grep " Delta: " | cut -d ":" -f2 );
SAMPLE_TIME=$(cat $OUTPUT_FILE | grep " Sample Time \=" | cut -d "=" -f2 );
RANGE=$(cat $OUTPUT_FILE | grep " Dynamic Range \=" | cut -d "=" -f2 | sed 's/{//' | sed 's/}//' | tr ',' ' ');
VERIFICATION=$(cat $OUTPUT_FILE | grep "VERIFICATION " | cut -d " " -f2 );
XSIZE=$(cat $OUTPUT_FILE | grep " X Size \=" | cut -d "=" -f2 );

SPACE=" ";
ENTER="\n";
ZERO="0";

ITEM=$OUTPUT_FILE$ENTER$REALIZATION$ENTER$IMPLEMENTATION$ENTER$NSIZE$ENTER$DSIZE$ENTER$NUMERATOR$ENTER$DENOMINATOR$ENTER$DELTA$ENTER$SAMPLE_TIME$ENTER$RANGE$ENTER$SPACE$VERIFICATION$ENTER$XSIZE$ENTER;

EXTRACTION="$EXTRACTION $ITEM";
fi

done


echo $EXTRACTION > dsv_counterexample_parameters.txt;
