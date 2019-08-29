#!/bin/bash
for threshold in 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8
do
    python findthres_predicting.py --no-pai True --source-file-name predicts-raw.json --threshold $threshold --result-file-name "predicts-threshold$threshold.json"
    cd ../metric
    python mrc_eval.py "predicts-threshold$threshold.json" ref.json v1 >> log_threshold
    cd ../predict
done
