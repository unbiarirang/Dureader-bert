# DuReader BERT

2019 DuReader 机器阅读理解模型。

reference code: [Dureader-Bert](https://github.com/basketballandlearn/Dureader-Bert)

预训练模型下载: [BERT-base-chinese](https://github.com/google-research/bert), [wwm & wwm-ext]( https://github.com/ymcui/Chinese-BERT-wwm)

DuReader数据下载: [DuReader_v2.0_preprocessed.zip](http://ai.baidu.com/broad/download?dataset=dureader)

## Code

- handle_data文件夹是处理DuReader的数据，与DuReader有关，与bert没有多大关系。
- dataset文件夹是处理中文数据的代码，大致是将文字转化为bert的输入: (inputs_ids,token_type_ids,input_mask)，然后做成dataloader。
- predict文件夹是用来预测的，基本与训练时差不多，一些细节不一样（输出）。
- 总的来说，只要输入符合bert的输入: (inputs_ids,token_type_ids,input_mask)就可以了。

## How to Run

**Dependencies**

- python3
- torch 1.0
- packages: pytorch-pretrained-bert, tqdm, torchtext

**Installation with pip**

```bash
pip install -r requirements.txt
```

**Preprocess the data**

将下载的 DuReader 数据放在data文件夹下。

```
|- data
| |- trainset
| | |- search.train.json
| | |- zhidao.train.json
| |- devset
| | |- search.dev.json
| | |- zhidao.dev.json
| |- testset
| | |- search.test.json
| | |- zhidao.test.json
```

```bash
# 数据处理
cd handle_data && sh run.sh
# 制作dataset
cd dataset && python run_squad.py
# 制作预测dataset
cd predict && python util.py
```

制作更多dataset

```bash
# 制作 qp-relevance 预测dataset
cd handle_data && sh run_qp.sh && cd ../predict && python util.py --dev-search-input-file '../../data/extracted/devset/search-qp.dev.json' --dev-zhidao-input-file '../../data/extracted/devset/zhidao-qp.dev.json' --predict-example-files 'predict-qp.data'
```

```bash
# 制作 no-match-score trainset
cd dataset && python run_squad_no_match_score.py
```

```bash
# 制作 synonym trainset（同义词替换训练集）
cd dataset && python run_squad_synonym.py
```

**Train**

```bash
python train.py --model-name 'best_model'
```

**Predict**

predict front 5 paragraphs:

```bash
cd predict && python predicting.py --model-name 'best_model' --result-file-name 'best_model.json'
```

predict top 5 qp-relevance score paragraphs :

```bash
cd predict && python predicting.py --model-name 'best_model' --result-file-name 'best_model-qp.json' --source-file-name predict-qp.data
```

ensemble predicting:

```bash
cd predict && python ensemble-predicting.py --model-names '["best_model1", "best_model2", "best_model3"]' --model-nums '[6, 6, 6]' --config-names '["bert_config.json", "bert_config.json", "bert_config.json"]' --result-file-name 'ensemble-qp.json' --source-file-name predict-qp.data
```

**Eval**

```bash
cd metric && python mrc_eval.py best_model.json ref.json v1
```

**All-in-one (train, predict, eval)**

```bash
sh train_and_predict.sh 8 2 512 3e-05 4 6
```

**Reproduce the best result**

train-synonym.data 1ep + train-no-match-score.data 2ep

```bash
python train.py --epochs 1 --model-name 'best_model_synonym' --trainset-name train-synonym.data --test-lines 186139 --state-dict pytorch_model_wwm_ext.bin --model-num 6
&& python train.py --model-name 'best_model_synonym' --state-dict model_dir/best_model_synonym --model-num 6 --trainset-name train-no-match-score.data --test-lines 229345
&& cd predict
&& python predicting.py --model-name 'best_model_synonym' --result-file-name 'best_model_synonym-qp.json' --source-file-name predict-qp.data --model-num 6 && cd ../metric
&& python mrc_eval.py best_model_synonym-qp.json ref.json v1
```

