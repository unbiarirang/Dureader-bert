import os
import json
import torch
import pickle
import argparse
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import predict_data
from tokenization import BertTokenizer
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modeling import BertConfig

# parse_args
parser = argparse.ArgumentParser()
parser.add_argument('--model-name', required=False, default='best_model')
parser.add_argument('--model-num', required=False, type=int, default=0)
parser.add_argument('--source-file-name', required=False, default='predict.data')
parser.add_argument('--result-file-name', required=False, default='predicts.json')
parser.add_argument('--config-name', required=False, default='bert_config.json')
parser.add_argument('--max-seq-length', required=False, type=int, default=512)
parser.add_argument('--max-query-length', required=False, type=int, default=60)
parser.add_argument('--max-answer-length', required=False, type=int, default=452)
parser.add_argument('--max-para-num', required=False, type=int, default=5)
parser.add_argument('--threshold', required=False, type=float, default=0.3)
args = parser.parse_args()

if args.model_num == 1:
    from modeling import BertForQuestionAnswering1 as BertForQuestionAnswering
elif args.model_num == 2:
    from modeling import BertForQuestionAnswering2 as BertForQuestionAnswering
elif args.model_num == 3:
    from modeling import BertForQuestionAnswering3 as BertForQuestionAnswering
elif args.model_num == 4:
    from modeling import BertForQuestionAnswering4 as BertForQuestionAnswering
elif args.model_num == 5:
    from modeling import BertForQuestionAnswering5 as BertForQuestionAnswering
elif args.model_num == 6:
    from modeling import BertForQuestionAnswering6 as BertForQuestionAnswering
else: # default model
    from modeling import BertForQuestionAnswering

model_dir = '../model_dir'
MODEL_PATH = os.path.join(model_dir, args.model_name)
CONFIG_PATH = os.path.join(model_dir, args.config_name)

def find_best_answer_for_passage(start_probs, end_probs):
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

def find_best_indexes_for_passage(start_probs, end_probs, passage_len, question):
    best_start_idxs, best_end_idxs = [], [] # (idx, prob)
    best_start, best_end, max_prob = -1, -1, 0

    start_probs, end_probs = start_probs.unsqueeze(0), end_probs.unsqueeze(0)
    start_probs, end_probs = F.softmax(start_probs, 1), F.softmax(end_probs, 1)
    num = 0
    while True:
        if num >= 5:
            break
        prob_start, best_start = torch.max(start_probs, 1)
        prob_end, best_end = torch.max(end_probs, 1)
        if best_end <= best_start or best_start <= len(question):
            num += 1
            continue
        if start_probs[0][best_start] == 0.0 or end_probs[0][best_end] == 0.0:
            num += 1
            continue

        best_start_idxs.append((best_start.item(), start_probs[0][best_start].item()))
        best_end_idxs.append((best_end.item(), end_probs[0][best_end].item()))
        start_probs[0][best_start], end_probs[0][best_end] = 0.0, 0.0
        num += 1


    candidates = []
    for best_start in best_start_idxs:
        for best_end in best_end_idxs:
            score = best_start[1] + best_end[1]
            if score > args.threshold:
                candidates.append({'start_idx': best_start[0], 'end_idx': best_end[0], 'score': score})
    # we need at least one answer span
    if len(candidates) == 0:
        candidates.append({'start_idx': best_start_idxs[0][0], 'end_idx': best_end_idxs[0][0], 'score': best_start_idxs[0][1] + best_end_idxs[0][1]})

    return candidates

def find_multiple_answers(sample, start_probs, end_probs, prior_scores=(0.44, 0.23, 0.15, 0.09, 0.07)):

    best_p_idx, best_span, best_score = None, None, 0
    all_candidates = []
    for p_idx, passage in enumerate(sample['doc_tokens'][:args.max_para_num]):

        passage_len = min(args.max_seq_length, len(passage['doc_tokens']))
        answer_span, score = find_best_answer_for_passage(start_probs[p_idx], end_probs[p_idx])
        candidates = find_best_indexes_for_passage(start_probs[p_idx], end_probs[p_idx], passage_len, sample['question_text'])

        answer = "p" + sample['question_text'] + "。" + sample['doc_tokens'][p_idx]['doc_tokens']
        for candidate in candidates:
            candidate['answer'] = answer[candidate['start_idx']: candidate['end_idx']+1]
            candidate['p_idx'] = p_idx
        all_candidates += candidates

    sorted(all_candidates, reverse=True, key=lambda x: x['score'])

    final_answers = []
    final_answer_str = ''
    for candidate in all_candidates:
        duplicated = False
        c_start_idx = candidate['start_idx']
        c_end_idx = candidate['end_idx']

        for final_answer in final_answers:
            a_start_idx = final_answer['start_idx']
            a_end_idx = final_answer['end_idx']

            # check if the answer is duplicated
            if candidate['p_idx'] == final_answer['p_idx']:
                if (c_start_idx >= a_start_idx and c_start_idx <= a_end_idx) or (c_end_idx >= a_start_idx and c_end_idx <= a_end_idx):
                    duplicated = True
                    break

        if len(final_answer_str) + len(candidate['answer']) > args.max_answer_length:
            break

        if not duplicated:
            final_answers.append(candidate)
            final_answer_str += candidate['answer']

    return final_answer_str[:args.max_answer_length]

def find_best_answer(sample, start_probs, end_probs, prior_scores=(0.44, 0.23, 0.15, 0.09, 0.07)):

    best_p_idx, best_span, best_score = None, None, 0

    for p_idx, passage in enumerate(sample['doc_tokens'][:args.max_para_num]):

        passage_len = min(args.max_seq_length, len(passage['doc_tokens']))
        answer_span, score = find_best_answer_for_passage(start_probs[p_idx], end_probs[p_idx])

        score *= prior_scores[p_idx]
        answer = "p" + sample['question_text'] + "。" + sample['doc_tokens'][p_idx]['doc_tokens']
        answer = answer[answer_span[0]: answer_span[1]+1]

        if score > best_score:
            best_score = score
            best_p_idx = p_idx
            best_span = answer_span

    if best_p_idx is None or best_span is None:
        best_answer = ''
    else:
        para = "p" + sample['question_text'] + "。" + sample['doc_tokens'][best_p_idx]['doc_tokens']
        best_answer = ''.join(para[best_span[0]: best_span[1]+1])

    return best_answer, best_p_idx

def evaluate(model, result_file):
    with open(args.source_file_name,'rb') as f:
        eval_examples = pickle.load(f)

    with torch.no_grad():
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        pred_answers, ref_answers = [], []

        for step, example in enumerate(tqdm(eval_examples)):
            start_probs, end_probs = [], []
            question_text = example['question_text']

            if len(example['doc_tokens']) != 0:
                for p_num, doc_tokens in enumerate(example['doc_tokens'][:args.max_para_num]):
                    (input_ids, input_mask, segment_ids) = \
                        predict_data.predict_data(question_text, doc_tokens['doc_tokens'], tokenizer, args.max_seq_length, args.max_query_length)
                    (input_ids, input_mask, segment_ids) = \
                        input_ids.to(device), input_mask.to(device), segment_ids.to(device)

                    start_prob, end_prob = model(input_ids, segment_ids, attention_mask=input_mask)     # !!!!!!!!!!
                    start_prob = start_prob[0]
                    end_prob = end_prob[0]
                    start_probs.append(start_prob.squeeze(0))
                    end_probs.append(end_prob.squeeze(0))
                best_answer, docs_index = find_best_answer(example, start_probs, end_probs)
#                best_answer = find_multiple_answers(example, start_probs, end_probs)
            else:
                best_answer = ''

            pred_answer = {'question_id': example['id'],
                                 'question':example['question_text'],
                                 'question_type': example['question_type'],
                                 'answers': [best_answer],
                                 'entity_answers': [[]],
                                 'yesno_answers': []}
            pred_answers.append(pred_answer)
            if 'answers' in example:
                ref_answers.append({'question_id': example['id'],
                                    'question_type': example['question_type'],
                                    'answers': example['answers'],
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
    output_config_file = CONFIG_PATH

    config = BertConfig(output_config_file)
    model = BertForQuestionAnswering(config)
    if next(model.parameters()).is_cuda:
        try:
            model.load_state_dict(torch.load(output_model_file))
        except:
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(output_model_file))
    else:
        try:
            model.load_state_dict(torch.load(output_model_file, map_location='cpu'))
        except:
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(output_model_file, map_location='cpu'))

    result_file_path = os.path.join('../metric', args.result_file_name)
    evaluate(model, result_file=result_file_path)

eval_all()
