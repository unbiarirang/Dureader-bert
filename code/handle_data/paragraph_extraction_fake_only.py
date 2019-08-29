#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
import json
import copy
from preprocess import metric_max_over_ground_truths, f1_score, find_best_question_match

def compute_paragraph_score(sample):
    """
    For each paragraph, compute the f1 score compared with the question
    Args:
        sample: a sample in the dataset.
    Returns:
        None
    Raises:
        None
    """
    scores = []
    question = sample["segmented_question"]
    for doc in sample['documents']:
        doc['segmented_paragraphs_scores'] = []
        for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):
            if len(question) > 0:
                related_score = metric_max_over_ground_truths(f1_score,
                        para_tokens,
                        [question])
            else:
                related_score = 0.0
            doc['segmented_paragraphs_scores'].append(related_score)

            scores.append(related_score)
#    with open('scores.txt', 'a') as f:
 #       f.write(str(scores) + '\n' + '\n')
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
    if 'most_related_para' in doc:
        para_id = doc['most_related_para']
    else:
        para_id = find_best_question_match(doc, question)

    doc['paragraphs_length'] = []
    for p_idx, segmented_paragraph in \
        enumerate(doc["segmented_paragraphs"]):
        doc['paragraphs_length'].append(len(segmented_paragraph))
        paragraph = ''.join(segmented_paragraph)
        if paragraph in paragraphs_his:
            del_ids.append(p_idx)
            if p_idx == para_id:
                para_id = paragraphs_his[paragraph]
            continue
        paragraphs_his[paragraph] = p_idx

    return False


def paragraph_selection(sample, mode, no_answer_cnt, invalid_cnt):
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

    doc_id = None
    para_id = None
    if 'answer_docs' in sample and len(sample['answer_docs']) > 0:
        doc_id = sample['answer_docs'][0]
        if doc_id >= len(sample['documents']):
            # Data error, answer doc ID > number of documents, this sample
            # will be filtered by dataset.py
            invalid_cnt += 1
            return False
        para_id = sample['documents'][doc_id]['most_related_para']
        if para_id >= len(sample['documents'][doc_id]['segmented_paragraphs']):
            # Data error, answer para ID > number of paragraphs, this sample
            # will be filtered by dataset.py
            invalid_cnt += 1
            return False
    else:
        no_answer_cnt += 1
        return False

    doc = sample['documents'][doc_id]

    status = dup_remove(doc, sample['question'])
    segmented_title = doc["segmented_title"]
    title_len = len(segmented_title)
    total_len = title_len + sum(doc['paragraphs_length'])
    # add splitter
    para_num = len(doc["segmented_paragraphs"])
    total_len += para_num
    incre_len = title_len + 1
    total_segmented_content = copy.deepcopy(segmented_title)
    total_segmented_content += [splitter] + doc["segmented_paragraphs"][para_id]
    answer_start = incre_len + sample['answer_spans'][0][0]
    answer_end = incre_len + sample['answer_spans'][0][1]
    sample['answer_spans'][0][0] = answer_start
    sample['answer_spans'][0][1] = answer_end
    doc["segmented_paragraphs"] = [total_segmented_content]
    doc["segmented_paragraphs_scores"] = [1.0]
    doc['paragraphs_length'] = [total_len]
    doc['paragraphs'] = [''.join(total_segmented_content)]
    doc['most_related_para'] = 0

    # delete all other documents
    sample['documents'] = doc
    return True


if __name__ == "__main__":
    # mode="train"/"dev"/"test"
    mode = sys.argv[1]
    no_answer_cnt = 0
    invalid_cnt = 0
    for line in sys.stdin:
        line = line.strip()
        if line == "":
            continue
        try:
            sample = json.loads(line, encoding='utf8')
        except:
            print >>sys.stderr, "Invalid input json format - '{}' will be ignored".format(line)
            continue

        is_valid = paragraph_selection(sample, mode, no_answer_cnt, invalid_cnt)
        if is_valid:
            print(json.dumps(sample, ensure_ascii=False))

