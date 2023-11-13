#!/bin/bash
python jsonl_to_json.py --in_file ${1} --out_file tmp.json --test
python test.py --validation_file tmp.json --model_name_or_path final_model
python json_to_jsonl.py --in_file prediction.json --out_file ${2}
