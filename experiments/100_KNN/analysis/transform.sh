#!/bin/bash

exec < Rational_Polypharm_Kinase_MAXIS_Frequencies.csv
read kinases
read freq
for i in {1..190}; do
	echo $(awk -F "," "{print \$$i}" top.csv),$(awk -F "," "{print \$$i}" bottom.csv)
done	
