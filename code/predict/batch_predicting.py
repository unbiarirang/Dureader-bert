import os
import json
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import predict_data
from tokenization import BertTokenizer
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modeling import BertForQuestionAnswering, BertConfig

import hdfs

# parse_args
parser = argparse.ArgumentParser()
parser.add_argument('--no-pai', required=False, type=bool, default=False)
parser.add_argument('--batch-size', required=False, type=int, default=2)
parser.add_argument('--model-name', required=False, default='best_model')
parser.add_argument('--source-file-name', required=False, default='./predict.data')
parser.add_argument('--result-file-name', required=False, default='predicts.json')
parser.add_argument('--config-name', required=False, default='bert_config.json')
parser.add_argument('--max-seq-length', required=False, type=int, default=512)
parser.add_argument('--max-query-length', required=False, type=int, default=60)
parser.add_argument('--max-para-num', required=False, type=int, default=5)
args = parser.parse_args()

model_dir = '../model_dir'

# PAI settings
if not args.no_pai:
    client = hdfs.InsecureClient("http://10.151.40.179", user="thsi_yicui")
    hdfs.client._Request.webhdfs_prefix = "/webhdfs/api/v1"
    pai_file_input = "/Container/thsi_yicui/dureader-bert/Dureader/output"
    local_file = "output"
    client.download(pai_file_input, local_file)
    model_dir = local_file

MODEL_PATH = os.path.join(model_dir, args.model_name)

class PredictSet(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        sample = self.samples[index]
        while len(sample['doc_tokens']) < args.max_para_num:
            sample['doc_tokens'].append({'doc_tokens': ''})
        while len(sample['answers']) < args.max_para_num:
            sample['answers'].append('')
        return sample

def find_best_answer_for_passage(start_probs, end_probs, passage_len, question):
    best_start, best_end, max_prob = -1, -1, 0

    start_probs, end_probs  = start_probs.unsqueeze(0), end_probs.unsqueeze(0)
    prob_start, best_start = torch.max(start_probs, 1)
    prob_end, best_end = torch.max(end_probs, 1)
    num = 0
    while True:
        if num > 3:
            break
        if best_end >= best_start:
            break
        else:
            start_probs[0][best_start], end_probs[0][best_end] = 0.0, 0.0
            prob_start, best_start = torch.max(start_probs, 1)
            prob_end, best_end = torch.max(end_probs, 1)
        num += 1
    max_prob = prob_start * prob_end

    if best_start <= best_end:
        return (best_start, best_end), max_prob
    else:
        return (best_end, best_start), max_prob

def find_best_answer(sample, start_probs, end_probs, prior_scores=(0.44, 0.23, 0.15, 0.09, 0.07)):

    best_p_idx, best_span, best_score = None, None, 0

    for p_idx, passage in enumerate(sample['doc_tokens'][:args.max_para_num]):
        passage_len = min(args.max_seq_length, len(passage))
        if passage_len == 0:
            continue
        answer_span, score = find_best_answer_for_passage(start_probs[p_idx], end_probs[p_idx], passage_len, sample['question_text'])

        score *= prior_scores[p_idx]

        answer = "p" + sample['question_text'] + "。" + sample['doc_tokens'][p_idx]
        answer = answer[answer_span[0]: answer_span[1]+1]

        if score > best_score:
            best_score = score
            best_p_idx = p_idx
            best_span = answer_span

    if best_p_idx is None or best_span is None:
        best_answer = ''
    else:
        para = "p" + sample['question_text'] + "。" + sample['doc_tokens'][best_p_idx]
        best_answer = ''.join(para[best_span[0]: best_span[1]+1])

    return best_answer, best_p_idx

def evaluate(model, result_file):
    with open(args.source_file_name,'rb') as f:
        eval_samples = pickle.load(f)

    params = {'batch_size' : args.batch_size,
              'shuffle'    : False,
              'num_workers': 1}
    predict_set = PredictSet(eval_samples)
    predict_generator = DataLoader(predict_set, **params)

    with torch.no_grad():
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        pred_answers, ref_answers = [], []

        for step, batch in enumerate(tqdm(predict_generator)):
            start_probss, end_probss = [], []
            samples = []
            question_ids = batch['id']
            question_types = batch['question_type']
            question_texts = batch['question_text']
            doc_tokenss = batch['doc_tokens']
            answerss = batch['answers']
            for id, type, text in zip(question_ids, question_types, question_texts):
                samples.append({'id': id, 'question_type': type, 'question_text': text,
                                'doc_tokens': [], 'answers': []})

            start_probss, end_probss = [], []
            for i in range(0, args.max_para_num):
                input_idss, input_masks, segment_idss = [], [], []
                b_doc_tokens = doc_tokenss[i]['doc_tokens']
                b_answers = answerss[i]

                for idx, (doc_tokens, question_text) in enumerate(zip(b_doc_tokens, question_texts)):
                    samples[idx]['doc_tokens'].append(doc_tokens)
                    if doc_tokens != []:
                        (input_ids, input_mask, segment_ids) = \
                            predict_data.predict_data(question_text, doc_tokens, tokenizer, args.max_seq_length, args.max_query_length)
                        (input_ids, input_mask, segment_ids) = \
                            input_ids.to(device), input_mask.to(device), segment_ids.to(device)
                        input_idss.append(input_ids)
                        input_masks.append(input_mask)
                        segment_idss.append(segment_ids)
                    else:
                        input_idss.append(torch.zeros([args.max_seq_length]))
                        input_masks.append(torch.zeros([args.max_seq_length]))
                        segment_idss.append(torch.zeros([args.max_seq_length]))
                for idx, answers in enumerate(b_answers):
                    samples[idx]['answers'].append(answers)

                # get padding
                pad_input_idss = [torch.nn.functional.pad(input_ids, (0, args.max_seq_length - input_ids.size()[-1])).squeeze(0) for input_ids in input_idss]
                pad_segment_idss = [torch.nn.functional.pad(segment_ids, (0, args.max_seq_length - segment_ids.size()[-1]), value=1).squeeze(0) for segment_ids in segment_idss]
                pad_input_masks = [torch.nn.functional.pad(input_mask, (0, args.max_seq_length - input_mask.size()[-1])).squeeze(0) for input_mask in input_masks]
                pad_input_idss = torch.stack(pad_input_idss)
                pad_segment_idss = torch.stack(pad_segment_idss)
                pad_input_masks = torch.stack(pad_input_masks)

                # infer
                start_probs, end_probs = model(pad_input_idss, pad_segment_idss, attention_mask=pad_input_masks)
                start_probss.append(start_probs)
                end_probss.append(end_probs)
            start_probss = torch.stack(start_probss)
            end_probss = torch.stack(end_probss)
            start_probss = start_probss.transpose(0,1).squeeze(-1)
            end_probss = end_probss.transpose(0,1).squeeze(-1)

            for (start_probs, end_probs, sample) in zip(start_probss, end_probss, samples):
                best_answer, docs_index = find_best_answer(sample, start_probs, end_probs)
                pred_answers.append({'question_id': sample['id'].item(),
                                     'question':sample['question_text'],
                                     'question_type': sample['question_type'],
                                     'answers': [best_answer],
                                     'entity_answers': [[]],
                                     'yesno_answers': []})
                if 'answers' in sample:
                    ref_answers.append({'question_id': sample['id'].item(),
                                        'question_type': sample['question_type'],
                                        'answers': sample['answers'],
                                        'entity_answers': [[]],
                                        'yesno_answers': []})
        with open(result_file, 'w', encoding='utf-8') as fout:
            for pred_answer in pred_answers:
                fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')
            print('save predicted data: ', result_file)
        with open("../metric/ref.json", 'w', encoding='utf-8') as fout:
            for pred_answer in ref_answers:
                fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')

def eval_all():

#    output_model_file = "../../output/best_model"
    output_model_file = MODEL_PATH
    output_config_file = os.path.join('../model_dir/', args.config_name)

    config = BertConfig(output_config_file)
    model = BertForQuestionAnswering(config)
    if not args.no_pai:
        try:
            model.load_state_dict(torch.load(output_model_file))#, map_location='cpu'))
        except:
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(output_model_file))#, map_location='cpu'))
    else:
        try:
            model.load_state_dict(torch.load(output_model_file, map_location='cpu'))
        except:
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(output_model_file, map_location='cpu'))

    result_file_path = os.path.join('../metric', args.result_file_name)
    evaluate(model, result_file=result_file_path)
    if not args.no_pai:
        print(os.getcwd())
        pai_file_output = "/Container/thsi_yicui/dureader-bert/Dureader/output"
        client.upload(pai_file_output, result_file_path, overwrite=True)

eval_all()
