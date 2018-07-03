#!/bin/bash

notifyme(){
		start=$(date +%s)
		"$@"
		notify-send "I'm Finished!" "\"$(echo $@)\" took $(($(date +%s) - start)) seconds to finish."
		paplay ~/.local/sndfiles/ding_ding.wav
}

# $1 should be the output directory
rm $1/*

notifyme python cpso.py --npart 10 --ndim 190 -c 2.1 2.1 2.1 --terms accuracy sensitivity \
specificity low_number overfitting --weights 0.2 0.2 0.2 0.2 0.2 --xbounds -6.0 \
6.0 --vbounds -3.0 1.0 --wbounds 0.4 0.9 -t 20 \
--data working_data/prepped_for_classifier/data.csv \
--target working_data/prepped_for_classifier/target.csv \
--labels working_data/prepped_for_classifier/feature_labels.csv \
--expname TESTING --outpath $1 --initpart
