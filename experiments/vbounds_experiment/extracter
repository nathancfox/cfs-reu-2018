echo "index,vmin,vmax,num_of_features,a_fitness,training_accuracy,test_accuracy"
for dir in *vbounds*
do
	line=$(echo $dir | awk 'BEGIN {FS="_"}{print $1 "," $3 "," $4}')
	line="$line,$(cat $dir/summary_results.out | grep "Number of Features" | awk 'BEGIN {FS=": "}{print $2}')"
	line="$line,$(cat $dir/summary_results.out | grep "Fitness (a_fitness" | awk 'BEGIN {FS=": "}{print $2}')"
	line="$line,$(cat $dir/summary_results.out | grep "TRAINING DATA" | awk 'BEGIN {FS=": "}{print $2}')"
	line="$line,$(cat $dir/summary_results.out | grep "TEST DATA" | awk 'BEGIN {FS=": "}{print $2}')"
	echo $line
done
