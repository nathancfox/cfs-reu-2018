#!/bin/bash

echo $(head -n1 00_iteration/var_by_time.csv)
for dir in [^c]*iteration*
do
	echo $(tail -n1 $dir/var_by_time.csv)
done
