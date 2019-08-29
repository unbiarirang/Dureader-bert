import re
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--file-name1', default='devset/search.dev.json')
parser.add_argument('-b', '--file-name2', default='devset/zhidao.dev.json')
parser.add_argument('-r', '--result-name', default='qp.dev.tsv')
args = parser.parse_args()

qp_inputs = []

def read_file(file_name):
    with open(file_name, 'r') as f:
        for line in f.readlines():
            sample = json.loads(line)
            question = sample['question']
            question_id = sample['question_id']
            docs = sample['documents']
            for doc_idx, doc in enumerate(docs):
                paras = doc['paragraphs']
                for para_idx, para in enumerate(paras):
                    qp_inputs.append((question, para, question_id, doc_idx, para_idx))

def main():
    read_file(args.file_name1)
    read_file(args.file_name2)
    with open(args.result_name, 'w') as f:
        for qp_input in qp_inputs:
            f.write(re.sub(r'\t', ' ', qp_input[0]) + '\t' + re.sub(r'\t', ' ', qp_input[1]) + '\t' + str(qp_input[2]) + '\t' + str(qp_input[3]) + '\t' + str(qp_input[4]) + '\n')

if __name__ == '__main__':
    main()
