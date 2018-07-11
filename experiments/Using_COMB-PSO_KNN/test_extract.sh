#!/bin/bash

awk '/Selected Features/{flag=1; next} /Runtime/{flag=0} flag' summary_results.out | tr -d ' ' > test_out.txt
sed -i '1d;$d' test_out.txt
sed ':a;$!N;s/\n/,/;ta' test_out.txt > final_out.txt
