#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
import json
import copy
from preprocess import metric_max_over_ground_truths, f1_score, find_best_question_match

def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf8") as f:
        # modify by yz
        # reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        data = f.readlines()
        lines = []
        for line in data:
            line = line.replace("\0", '').rstrip()
            split_line = line.split('\t')
            lines.append(split_line)
        return lines

def read_paragraph_score_qp(qp_file_name):
    lines = read_tsv(qp_file_name)
    qid_scores = {}
    for line in lines:
        question_id = int(line[2])
        doc_idx = int(line[3])
        para_idx = int(line[4])
        score = float(line[-1])
        if question_id in qid_scores:
            if doc_idx in qid_scores[question_id]:
                qid_scores[question_id][doc_idx][para_idx] = score
            else:
                qid_scores[question_id][doc_idx] = {para_idx: score}
        else:
            qid_scores[question_id] = {doc_idx: {para_idx: score}}

    return qid_scores

def compute_paragraph_score_qp(sample, qid_scores):
    scores = []
    question = sample["segmented_question"]
    question_id = sample["question_id"]
    for d_idx, doc in enumerate(sample['documents']):
        doc['segmented_paragraphs_scores'] = []
        for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):
            if len(question) > 0:
                try: ###FIXME
                    related_score = qid_scores[question_id][d_idx][p_idx]
                except:
                    related_score = 0.0
            else:
                related_score = 0.0
            doc['segmented_paragraphs_scores'].append(related_score)
            scores.append(related_score)

def dup_remove(doc, question):
    """
    For each document, remove the duplicated paragraphs
    Args:
        doc: a doc in the sample
    Returns:
        bool
    Raises:
        None
    """
    paragraphs_his = {}
    del_ids = []
    para_id = None
    if 'most_related_para' in doc and mode == 'train':
        para_id = doc['most_related_para']
    else:
        para_id = find_best_question_match(doc, question)

    doc['paragraphs_length'] = []
    for p_idx, (segmented_paragraph, paragraph_score) in \
        enumerate(zip(doc["segmented_paragraphs"], doc["segmented_paragraphs_scores"])):
        doc['paragraphs_length'].append(len(segmented_paragraph))
        paragraph = ''.join(segmented_paragraph)
        if paragraph in paragraphs_his:
            del_ids.append(p_idx)
            if p_idx == para_id:
                para_id = paragraphs_his[paragraph]
            continue
        paragraphs_his[paragraph] = p_idx
    # delete
    prev_del_num = 0
    del_num = 0
    for p_idx in del_ids:
        if p_idx < para_id:
            prev_del_num += 1
        del doc["segmented_paragraphs"][p_idx - del_num]
        del doc["segmented_paragraphs_scores"][p_idx - del_num]
        del doc['paragraphs_length'][p_idx - del_num]
        del_num += 1
    if len(del_ids) != 0:
        if 'most_related_para' in doc and mode == 'train':
            doc['most_related_para'] = para_id - prev_del_num
        doc['paragraphs'] = []
        for segmented_para in doc["segmented_paragraphs"]:
            paragraph = ''.join(segmented_para)
            doc['paragraphs'].append(paragraph)
        return True
    else:
        return False


def paragraph_selection(sample, mode):
    """
    For each document, select paragraphs that includes as much information as possible
    Args:
        sample: a sample in the dataset.
        mode: string of ("train", "dev", "test"), indicate the type of dataset to process.
    Returns:
        None
    Raises:
        None
    """
    scores = []
    # predefined maximum length of paragraph
    MAX_P_LEN = 510
    # predefined splitter
  #  splitter = u'<splitter>'
    splitter = '。'

    # topN of related paragraph to choose
    topN = 5
    doc_id = None
    if 'answer_docs' in sample and len(sample['answer_docs']) > 0 and mode == 'train':
        doc_id = sample['answer_docs'][0]
        if doc_id >= len(sample['documents']):
            # Data error, answer doc ID > number of documents, this sample
            # will be filtered by dataset.py
            return
    for d_idx, doc in enumerate(sample['documents']):
        if 'segmented_paragraphs_scores' not in doc:
            continue
        status = dup_remove(doc, sample['question'])
        segmented_title = doc["segmented_title"]
        title_len = len(segmented_title)
        para_id = None
        if doc_id is not None and mode == 'train':
            para_id = sample['documents'][doc_id]['most_related_para']
        total_len = title_len + sum(doc['paragraphs_length'])
        # add splitter
        para_num = len(doc["segmented_paragraphs"])
        total_len += para_num
        if total_len <= MAX_P_LEN:
            incre_len = title_len
            total_segmented_content = copy.deepcopy(segmented_title)
            for p_idx, segmented_para in enumerate(doc["segmented_paragraphs"]):
                if doc_id == d_idx and para_id > p_idx:
                    incre_len += len([splitter] + segmented_para)
                if doc_id == d_idx and para_id == p_idx:
                    incre_len += 1
                total_segmented_content += [splitter] + segmented_para
            if doc_id == d_idx:
                answer_start = incre_len + sample['answer_spans'][0][0]
                answer_end = incre_len + sample['answer_spans'][0][1]
                sample['answer_spans'][0][0] = answer_start
                sample['answer_spans'][0][1] = answer_end
            doc["segmented_paragraphs"] = [total_segmented_content]
            doc["segmented_paragraphs_scores"] = [1.0]
            doc['paragraphs_length'] = [total_len]
            doc['paragraphs'] = [''.join(total_segmented_content)]
            doc['most_related_para'] = 0
            continue
        # find topN paragraph id
        para_infos = []
        for p_idx, (para_tokens, para_scores) in \
                enumerate(zip(doc['segmented_paragraphs'], doc['segmented_paragraphs_scores'])):
            para_infos.append((para_tokens, para_scores, len(para_tokens), p_idx))

        para_infos.sort(key=lambda x: (-x[1], x[2]))
   #     para_infos.sort(key=lambda x: -x[1])
        topN_idx = []

        for para_info in para_infos[:topN]:
            topN_idx.append(para_info[-1])
            scores.append((para_info[1],para_info[2]))

        final_idx = []
        total_len = title_len
        if doc_id == d_idx:
            if mode == "train":
                final_idx.append(para_id)
                total_len = title_len + 1 + doc['paragraphs_length'][para_id]
        for id in topN_idx:
            if total_len > MAX_P_LEN:
                break
            if doc_id == d_idx and id == para_id and mode == "train":
                continue
            total_len += 1 + doc['paragraphs_length'][id]
            final_idx.append(id)
        total_segmented_content = copy.deepcopy(segmented_title)
        final_idx.sort()
        incre_len = title_len
        for id in final_idx:
            if doc_id == d_idx and id < para_id:
                incre_len += 1 + doc['paragraphs_length'][id]
            if doc_id == d_idx and id == para_id:
                incre_len += 1
            total_segmented_content += [splitter] + doc['segmented_paragraphs'][id]
        if doc_id == d_idx:
            answer_start = incre_len + sample['answer_spans'][0][0]
            answer_end = incre_len + sample['answer_spans'][0][1]
            sample['answer_spans'][0][0] = answer_start
            sample['answer_spans'][0][1] = answer_end
        doc["segmented_paragraphs"] = [total_segmented_content]
        doc["segmented_paragraphs_scores"] = [1.0]
        doc['paragraphs_length'] = [total_len]
        doc['paragraphs'] = [''.join(total_segmented_content)]
        doc['most_related_para'] = 0


if __name__ == "__main__":
    # mode="train"/"dev"/"test"
    mode = sys.argv[1]
    tsv_file_name = sys.argv[2]
    qid_scores = read_paragraph_score_qp(tsv_file_name)
    for line in sys.stdin:
        line = line.strip()
        if line == "":
            continue
        try:
            sample = json.loads(line, encoding='utf8')
        except:
            print >>sys.stderr, "Invalid input json format - '{}' will be ignored".format(line)
            continue
        compute_paragraph_score_qp(sample, qid_scores)
        paragraph_selection(sample, mode)
        print(json.dumps(sample, ensure_ascii=False))

