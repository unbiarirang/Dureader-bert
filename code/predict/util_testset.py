import json
import torch
import pickle
import argparse
from tqdm import tqdm

# parse_args
parser = argparse.ArgumentParser()
parser.add_argument('--test-search-input-file', required=False, default='../../data/extracted/testset/search.test.json')
parser.add_argument('--test-zhidao-input-file', required=False, default='../../data/extracted/testset/zhidao.test.json')
parser.add_argument('--predict-example-files', required=False, default='predict-test.data')
args = parser.parse_args()

def creat_examples(filename_1, filename_2, result):
    c1, c2, c3, c4, c5 = 0,0,0,0,0
    examples = []
    with open(filename_1, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            try:
                source = json.loads(line.strip())
            except:
                c1 += 1
                continue
            if not isinstance(source, dict):
                c2 += 1
                continue

            if 'doc_tokens' not in source:
                source['doc_tokens'] = []

            for doc in source['documents']:
                ques_len = len(doc['segmented_title']) + 1
                clean_doc = "".join(doc['segmented_paragraphs'][doc['most_related_para']][ques_len:])
                if len(clean_doc) > 4:
                    source['doc_tokens'].append( {'doc_tokens': clean_doc} )

            if len(source['documents']) == 0:
                print("error")
                print(source)
                c5 += 1
                continue

            example = ({
                        'id':source['question_id'],
                        'question_text':source['question'].strip(),
                        'question_type': source['question_type'],
                        'doc_tokens':source['doc_tokens'],
                       })
            examples.append(example)
        print(len(examples))
    with open(filename_2, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            try:
                source = json.loads(line.strip())
            except:
                c3 += 1
                continue
            if not isinstance(source, dict):
                c4 += 1
                continue

            if 'doc_tokens' not in source:
                source['doc_tokens'] = []

            for doc in source['documents']:
                ques_len = len(doc['segmented_title']) + 1
                clean_doc = "".join(doc['segmented_paragraphs'][doc['most_related_para']][ques_len:])
                if len(clean_doc) > 4:
                    source['doc_tokens'].append( {'doc_tokens': clean_doc} )

            if len(source['documents']) == 0:
                print("error")
                print(source)
                c5 += 1
            example = ({
                        'id':source['question_id'],
                        'question_text':source['question'].strip(),
                        'question_type': source['question_type'],
                        'doc_tokens':source['doc_tokens'],
                      })
            examples.append(example)
        print(len(examples))

    print("{} questions in total".format(len(examples)))
    print(c1,c2,c3,c4,c5)
    with open(result,'wb') as fw:
        pickle.dump(examples, fw)

if __name__ == "__main__":
    creat_examples(filename_1=args.test_zhidao_input_file,
                   filename_2=args.test_search_input_file,
                   result=args.predict_example_files     )

