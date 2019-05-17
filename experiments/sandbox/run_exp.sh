number=$1
python cpso.py --npart 100 --ndim 190 -c 2.0 1.5 2.5 -a 0.75 --testsize 0.33 --xbounds -6.0 6.0 --vbounds -2.0 4.0 --wbounds 0.4 1.5 -t 150 --data data/data.csv --target data/target.csv --labels data/feature_labels.csv --expname "Playing with Parameters - $number - Random Forest" --author "Nathan Fox" --outpath output/
cp output/summary_results.out reports/report_$number.out
python plotter.py $number
eog figures/plot_$number.svg 
