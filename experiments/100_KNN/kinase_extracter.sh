#!/bin/bash

for dir in 0*iteration
do
	awk '/Selected Features/{flag=1; next} /Runtime/{flag=0} flag' $dir/summary_results.out | tr -d ' ' > temp.txt
	sed -i '1d;$d' temp.txt
	sed ':a;$!N;s/\n/,/;ta' temp.txt
done
rm temp.txt
