echo "num_of_features,accuracy,sensitivity,specificity"
for dir in control_??_iteration
do
	line=190
	line="$line,$(cat $dir/summary_results.out | grep "Accuracy" | awk 'BEGIN {FS=": "}{print $2}')"
	line="$line,$(cat $dir/summary_results.out | grep "Sensitivity" | awk 'BEGIN {FS=": "}{print $2}')"
	line="$line,$(cat $dir/summary_results.out | grep "Specificity" | awk 'BEGIN {FS=": "}{print $2}')"
	echo $line
done
