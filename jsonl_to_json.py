import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--in_file", type=str, default="train.jsonl")
parser.add_argument("--out_file", type=str, default="train.json")
parser.add_argument("--test", action="store_true")
args = parser.parse_args()

def jsonl_to_json(jsonl_filename, json_filename):
    with open(jsonl_filename, 'r') as jsonl_file:
        # Read JSON Lines file line by line
        json_lines = [json.loads(line) for line in jsonl_file]

    # Write the list of JSON objects to a regular JSON file
    with open(json_filename, 'w', encoding='utf-8') as json_file:
        for line in json_lines:
            if args.test:
                line["title"] = ""
            line["maintext"] = line["maintext"].replace('\n', '')
        json.dump(json_lines, json_file, indent=2, ensure_ascii=False)

# Example usage
jsonl_to_json(args.in_file, args.out_file)
