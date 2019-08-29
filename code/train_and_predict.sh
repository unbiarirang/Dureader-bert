#!/bin/bash

batch_size=$1
epochs=$2
max_seq_length=${3:-512}
learning_rate=${4:-3e-05}
acc_steps=${5:-8}
model_num=${6:-0}

# output file names
if [ $model_num -eq 0 ]
then
    model_name="best_model-bs$batch_size-ep$epochs-msl$max_seq_length-lr$learning_rate-acc$acc_steps"
    result_name="predicts-bs$batch_size-ep$epochs-msl$max_seq_length-lr$learning_rate-acc$acc_steps.json"
else
    model_name="best_model-bs$batch_size-ep$epochs-msl$max_seq_length-lr$learning_rate-acc$acc_steps-model$model_num"
    result_name="predicts-bs$batch_size-ep$epochs-msl$max_seq_length-lr$learning_rate-acc$acc_steps-model$model_num.json"
fi

# get a max query length to a max sequence length
if [ $max_seq_length -eq 512 ]
then
    max_query_length=60
elif [ $max_seq_length -eq 115 ]
then
    max_query_length=15
elif [ $max_seq_length -eq 220 ] || [ $max_seq_length -eq 320 ] || [ $max_seq_length -eq 400 ]
then
    max_query_length=20
else
    echo ERROR: max_seq_length should be one of 512, 115, 220, 320, 400
    exit 1 # terminate and indicate error
fi
echo max query length: $max_query_length
echo max sequence length: $max_seq_length
echo learning rate: $learning_rate
echo model name: $model_name
echo predict result name: $result_name

# train
python train.py --batch-size $batch_size --epochs $epochs --model-num $model_num --max-seq-length $max_seq_length --learning-rate $learning_rate --gradient-accumulation-steps $acc_steps
# predict best model
cd predict
python predicting.py --model-name $model_name --result-file-name $result_name --model-num $model_num --max-seq-length $max_seq_length --max-query-length $max_query_length
# evaluate
cd ../metric
python mrc_eval.py $result_name ref.json v1
