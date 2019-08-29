import os
import torch
import random
import pickle
import argparse
from tqdm import tqdm
from torch import nn, optim

import evaluate
from optimizer import BertAdam
from dataset.dataloader import Dureader
from dataset.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
#from model_dir.modeling import BertForQuestionAnswering

# parse_args
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', required=False, type=int, default=8)
parser.add_argument('--epochs', required=False, type=int, default=2)
parser.add_argument('--max-seq-length', required=False, type=int, default=512)
parser.add_argument('--config-name', required=False, default='bert_config.json')
parser.add_argument('--model-name', required=False, default=None)
parser.add_argument('--dataset-path', required=False, default='../data/dataset')
parser.add_argument('--trainset-name', required=False, default=None)
parser.add_argument('--devset-name', required=False, default=None)
parser.add_argument('--state-dict', required=False, default=None)
parser.add_argument('--no-eval', required=False, default=False)
parser.add_argument('--model-num', required=False, type=int, default=0)
# 多少条训练数据，即：len(features), 记得修改 !!!!!!!!!!
parser.add_argument('--test-lines', required=False, type=int, default=186862)
parser.add_argument('--seed', required=False, type=int, default=42)
parser.add_argument('--save-dir', required=False, default='./model_dir')
# 选择几篇文档进行预测
parser.add_argument('--learning-rate', required=False, type=float, default=3e-5)
# 梯度累积
parser.add_argument('--gradient-accumulation-steps', required=False, type=int, default=4)
args = parser.parse_args()
print(args)

if args.model_num == 1:
    from model_dir.modeling import BertForQuestionAnswering1 as BertForQuestionAnswering
elif args.model_num == 2:
    from model_dir.modeling import BertForQuestionAnswering2 as BertForQuestionAnswering
elif args.model_num == 3:
    from model_dir.modeling import BertForQuestionAnswering3 as BertForQuestionAnswering
elif args.model_num == 4:
    from model_dir.modeling import BertForQuestionAnswering4 as BertForQuestionAnswering
elif args.model_num == 5:
    from model_dir.modeling import BertForQuestionAnswering5 as BertForQuestionAnswering
elif args.model_num == 6:
    from model_dir.modeling import BertForQuestionAnswering6 as BertForQuestionAnswering
else: # default model
    from model_dir.modeling import BertForQuestionAnswering

NUM_TRAIN_OPTIMIZATION_STEPS = int(args.test_lines / args.gradient_accumulation_steps / args.batch_size) * args.epochs
LOG_STEP = int(args.test_lines / args.batch_size / 4)  # 每个epoch验证几次，默认4次
if args.trainset_name is None:
    TRAINSET_NAME = 'train' + str(args.max_seq_length) + '.data'
else:
    TRAINSET_NAME = args.trainset_name
if args.devset_name is None:
    DEVSET_NAME = 'dev' + str(args.max_seq_length) + '.data'
else:
    DEVSET_NAME = args.devset_name
if args.model_name is None:
    SAVE_MODEL_NAME = 'best_model-bs' + str(args.batch_size) + '-ep' + str(args.epochs) \
                    + '-msl' + str(args.max_seq_length) + '-lr' + str(args.learning_rate) \
                    + '-acc' + str(args.gradient_accumulation_steps)
    SAVE_LAST_MODEL_NAME = 'last_model-bs' + str(args.batch_size) + '-ep' + str(args.epochs) \
                         + '-msl' + str(args.max_seq_length) + '-lr' + str(args.learning_rate) \
                         + '-acc' + str(args.gradient_accumulation_steps)
    if args.model_num != 0:
        SAVE_MODEL_NAME += '-modelnum' + str(args.model_num)
        SAVE_LAST_MODEL_NAME += '-modelnum' + str(args.model_num)
else:
    SAVE_MODEL_NAME = args.model_name
    SAVE_LAST_MODEL_NAME = 'last_' + args.model_name
SAVE_MODEL_PATH = os.path.join(args.save_dir, SAVE_MODEL_NAME)
SAVE_LAST_MODEL_PATH = os.path.join(args.save_dir, SAVE_LAST_MODEL_NAME)

print('TRAINSET_NAME: ', TRAINSET_NAME)
print('DEVSET_NAME: ', DEVSET_NAME)
print('SAVE_MODEL_NAME: ' , SAVE_MODEL_NAME)
print('SAVE_LAST_MODEL_NAME: ', SAVE_LAST_MODEL_NAME)

# random seed
random.seed(args.seed)
torch.manual_seed(args.seed)

def train():
    print(args.config_name)
    # 加载预训练bert
    if args.state_dict is not None:
        model = BertForQuestionAnswering.from_pretrained('bert-base-chinese',
                    cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                    'distributed_{}'.format(-1)),
                    state_dict=torch.load(args.state_dict),#, map_location='cpu'),
                    config_name = args.config_name)
#        try:
#            model.load_state_dict(torch.load(output_model_file))
#        except:
#            model = nn.DataParallel(model)
#            model.load_state_dict(torch.load(output_model_file))
    else:
        model = BertForQuestionAnswering.from_pretrained('bert-base-chinese',
                    cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                    'distributed_{}'.format(-1)),
                    config_name = args.config_name)

    # use multiple GPUs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print("We have ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    print('device: ', device)
    model.to(device)

    # 准备 optimizer
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=0.1, t_total=NUM_TRAIN_OPTIMIZATION_STEPS)

    # 准备数据
    data = Dureader(args.batch_size, args.dataset_path,
                    TRAINSET_NAME, DEVSET_NAME)
    train_dataloader, dev_dataloader = data.train_iter, data.dev_iter

    best_loss = 100000.0
    model.train()
    for i in range(args.epochs):
        print("==Epoch: ", i)
        for step , batch in enumerate(tqdm(train_dataloader, desc="Epoch")):
            input_ids, input_mask, segment_ids, start_positions, end_positions = \
                batch.input_ids, batch.input_mask, batch.segment_ids, batch.start_position, batch.end_position
            input_ids, input_mask, segment_ids, start_positions, end_positions = \
                input_ids.to(device), input_mask.to(device), segment_ids.to(device), start_positions.to(device), end_positions.to(device)

            # 计算loss
            loss, _, _ = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, start_positions=start_positions, end_positions=end_positions)
            loss = loss / args.gradient_accumulation_steps
            loss.sum().backward()

            # 更新梯度
            if (step+1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # 验证
            if step % LOG_STEP == 4 and not args.no_eval:
                eval_loss = evaluate.evaluate(model, dev_dataloader, device, args.gradient_accumulation_steps)
                print('eval loss: ', eval_loss)
                if eval_loss < best_loss:
                    print("save the best model")
                    best_loss = eval_loss
                    torch.save(model.state_dict(), SAVE_MODEL_PATH)
                    model.train()
    # 最后一次验证
    eval_loss = evaluate.evaluate(model, dev_dataloader, device, args.gradient_accumulation_steps)
    print('final eval loss: ', eval_loss)
    if eval_loss < best_loss:
        print("save the best model")
        best_loss = eval_loss
        torch.save(model.state_dict(), SAVE_MODEL_PATH)
        model.train()

    print('training finished!')

if __name__ == "__main__":
    train()
