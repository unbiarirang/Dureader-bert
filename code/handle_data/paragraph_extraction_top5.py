#!/usr/bin/python
#-*- coding:utf-8 -*-
# devset para extraction
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
    sample['segmented_paragraphs'] = []
    sample['segmented_paragraphs_scores'] = []
    for d_idx, doc in enumerate(sample['documents']):
        for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):
            if len(question) > 0:
                related_score = metric_max_over_ground_truths(f1_score,
                        para_tokens,
                        [question])
            else:
                related_score = 0.0
            sample['segmented_paragraphs'].append(para_tokens)
            sample['segmented_paragraphs_scores'].append(related_score)
            scores.append((related_score, d_idx, p_idx))

    with open('scores.txt', 'a') as f:
        f.write(str(scores) + '\n')

def dup_remove(doc):
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

    doc['paragraphs_length'] = []
    for p_idx, (segmented_paragraph, paragraph_score) in \
        enumerate(zip(doc["segmented_paragraphs"], doc["segmented_paragraphs_scores"])):
        doc['paragraphs_length'].append(len(segmented_paragraph))
        paragraph = ''.join(segmented_paragraph)
        if paragraph in paragraphs_his:
            del_ids.append(p_idx)
            continue
        paragraphs_his[paragraph] = p_idx
    # delete
    prev_del_num = 0
    del_num = 0
    for p_idx in del_ids:
        del doc["segmented_paragraphs"][p_idx - del_num]
        del doc["segmented_paragraphs_scores"][p_idx - del_num]
        del doc['paragraphs_length'][p_idx - del_num]
        del_num += 1
    if len(del_ids) != 0:
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

    # remove duplicated paragraphs
    for d_idx, doc in enumerate(sample['documents']):
        if 'segmented_paragraphs_scores' not in doc:
            continue
        status = dup_remove(doc)

    # find topN paragraph id over all documents
    para_infos = []
    for p_idx, (para_tokens, para_score) in \
            enumerate(zip(sample['segmented_paragraphs'], sample['segmented_paragraphs_scores'])):
        para_infos.append((para_tokens, para_score, p_idx))

    # sort paragraphs by score
    para_infos.sort(key=lambda x: (-x[1], x[2]))

    # store top5 in the first document
    d_idx = 0
    topN = 5
    total_len = 0
    total_segmented_content = []
    for para_info in para_infos[:topN]:
        para_tokens = para_info[0]
        para_len = len(para_tokens)
        para_score = para_info[1]
        para_idx = para_info[2]

        total_len +=  para_len
        total_segmented_content += para_tokens

    sample['documents'][d_idx]["segmented_paragraphs"] = [total_segmented_content]
    sample['documents'][d_idx]["segmented_paragraphs_scores"] = [1.0]
    sample['documents'][d_idx]['paragraphs_length'] = [total_len]
    sample['documents'][d_idx]['paragraphs'] = [''.join(total_segmented_content)]
    sample['documents'][d_idx]['most_related_para'] = 0
    del sample['segmented_paragraphs']
    del sample['segmented_paragraphs_scores']

    return


if __name__ == "__main__":
    # mode="train"/"dev"/"test"
    mode = sys.argv[1]
    for line in sys.stdin:
        line = line.strip()
        if line == "":
            continue
        try:
            sample = json.loads(line, encoding='utf8')
        except:
            print >>sys.stderr, "Invalid input json format - '{}' will be ignored".format(line)
            continue
        compute_paragraph_score(sample)
        paragraph_selection(sample, mode)
        print(json.dumps(sample, ensure_ascii=False))

