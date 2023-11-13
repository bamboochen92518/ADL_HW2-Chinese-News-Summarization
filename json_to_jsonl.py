import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--in_file", type=str, default="train.json")
parser.add_argument("--out_file", type=str, default="train.jsonl")
args = parser.parse_args()
# Read JSON data
with open(args.in_file, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Write JSON Lines
with open(args.out_file, 'w', encoding='utf-8') as jsonl_file:
    for item in data:
        jsonl_file.write(json.dumps(item, ensure_ascii=False) + '\n')
