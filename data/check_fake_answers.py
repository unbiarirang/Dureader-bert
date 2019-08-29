import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file-path', required=True)
args = parser.parse_args()

file_path = args.file_path

cnt = 0
cnt1 = 0
with open(file_path, 'r') as f:
    for line in f.readlines():
        sample = json.loads(line)
        paragraphs = ''
        for doc in sample['documents']:
            paragraphs += doc['paragraphs'][0]

        try:
            fake_answer = sample['fake_answers'][0]
        except:
            cnt1 += 1
            continue

        if fake_answer not in paragraphs:
            cnt += 1
print('fake answer is not in paragraphs: ', cnt)
print('no fake answer at all: ', cnt1)
