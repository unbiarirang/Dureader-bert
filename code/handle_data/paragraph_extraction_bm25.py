#!/usr/bin/python
#-*- coding:utf-8 -*-
import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
import json
import copy
import math
from preprocess import metric_max_over_ground_truths, f1_score, find_best_question_match

class BM25(object):
    def __init__(self, docs):
        self.D = len(docs)
        self.avgdl = sum([len(doc)+0.0 for doc in docs]) / self.D
        self.docs = docs
        self.f = []  # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
        self.df = {} # 存储每个词及出现了该词的文档数量
        self.idf = {} # 存储每个词的idf值
        self.k1 = 1.5
        self.b = 0.75
        self.init()

    def init(self):
        for doc in self.docs:
            tmp = {}
            for word in doc:
                tmp[word] = tmp.get(word, 0) + 1  # 存储每个文档中每个词的出现次数
            self.f.append(tmp)
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D-v+0.5)-math.log(v+0.5)

    def sim(self, doc, index):
        score = 0
        for word in doc:
            if word not in self.f[index]:
                continue
            d = len(self.docs[index])
            score += (self.idf[word]*self.f[index][word]*(self.k1+1)
                  / (self.f[index][word]+self.k1*(1-self.b+self.b*d
                                        / self.avgdl)))
        return score

    # 总共有N篇文档，传来的doc为查询文档，计算doc与所有文档匹配
    # 后的得分score，总共有多少篇文档，scores列表就有多少项，
    # 每一项为doc与这篇文档的得分，所以分清楚里面装的是文档得分，
    # 不是词语得分。
    def simall(self, doc):
        scores = []
        for index in range(self.D):
            score = self.sim(doc, index)
            scores.append(score)
        return scores

def find_best_question_match_bm25(doc, question, with_score=False):
    most_related_para = -1
    max_related_score = 0
    most_related_para_len = 0
    doc_ = doc['segmented_paragraphs']
    bm = BM25(doc_)
    scores = bm.simall(question)
    for p_idx, related_score in enumerate(scores):
        if related_score > max_related_score \
            or (related_score == max_related_score \
            and len(doc_[p_idx]) < most_related_para_len):
            most_related_para = p_idx
            max_related_score = related_score
            most_related_para_len = len(doc_[p_idx])

    if most_related_para == -1:
        most_related_para = 0
    if with_score:
        return most_related_para, max_related_score

    return most_related_para

def compute_question_paragraph_bm25_score(sample):
    question = sample['segmented_question']

    for doc in sample['documents']:
        bm = BM25(doc['segmented_paragraphs'])
        scores = bm.simall(question)
        doc['segmented_paragraphs_scores'] = scores

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
    if 'most_related_para' in doc and mode == 'train': ######FIXME
        para_id = doc['most_related_para']
    else:
        para_id = find_best_question_match_bm25(doc, question)

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
        if 'most_related_para' in doc and mode == 'train': #####FIXME
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
    # predefined maximum length of paragraph
    MAX_P_LEN = 510
    # predefined splitter
  #  splitter = u'<splitter>'
    splitter = '。'

    # topN of related paragraph to choose
    topN = 5
    doc_id = None
    if 'answer_docs' in sample and len(sample['answer_docs']) > 0 and mode == 'train': ######FIXME
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
        if doc_id is not None and mode == 'train': ######FIXME
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
        for p_idx, (para_tokens, para_score) in \
                enumerate(zip(doc['segmented_paragraphs'], doc['segmented_paragraphs_scores'])):
            para_infos.append((para_tokens, para_score, len(para_tokens), p_idx))

        para_infos.sort(key=lambda x: (-x[1], x[2]))
#        para_infos.sort(key=lambda x: -x[1])
        topN_idx = []
        scores = []

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
    for line in sys.stdin:
        line = line.strip()
        if line == "":
            continue
        try:
            sample = json.loads(line)
        except:
            print >>sys.stderr, "Invalid input json format - '{}' will be ignored".format(line)
            continue
#        compute_paragraph_score(sample)
        compute_question_paragraph_bm25_score(sample)
        paragraph_selection(sample, mode)
        print(json.dumps(sample,  ensure_ascii=False))

