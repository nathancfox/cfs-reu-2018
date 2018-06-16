echo "index,vcenter,vrange,vmin,vmax,num_of_features,a_fitness,training_score,test_score"
for dir in *vrange*
do
	line=$(echo $dir | awk 'BEGIN {FS="_"}{print $1 "," $3 "," $5}')
	line="$line,$(cat $dir/summary_results.out | grep "Velocity Bounds (v_bounds)" | awk '{print $5}' | sed 's/[(,]//' | sed 's/[(,]//')"
	line="$line,$(cat $dir/summary_results.out | grep "Velocity Bounds (v_bounds)" | awk '{print $6}' | sed 's/[)]//')"
	line="$line,$(cat $dir/summary_results.out | grep "Number of Features" | awk 'BEGIN {FS=": "}{print $2}')"
	line="$line,$(cat $dir/summary_results.out | grep "Fitness (a_fitness" | awk 'BEGIN {FS=": "}{print $2}')"
	line="$line,$(cat $dir/summary_results.out | grep "TRAINING DATA" | awk 'BEGIN {FS=": "}{print $2}')"
	line="$line,$(cat $dir/summary_results.out | grep "TEST DATA" | awk 'BEGIN {FS=": "}{print $2}')"
	echo $line
done
