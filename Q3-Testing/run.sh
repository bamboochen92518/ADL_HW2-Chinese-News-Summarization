#!/bin/bash

/tmp2/b10902005/adl_env/bin/python test.py --top_p 0.25 --output_file pred_p025.json --model_name_or_path epoch_9 --validation_file ./data/public.json
/tmp2/b10902005/adl_env/bin/python ../data/json_to_jsonl.py --in_file pred_p025.json --out_file pred_p025.jsonl
/tmp2/b10902005/adl_env/bin/python ../ADL23-HW2/eval.py -r ../data/public.jsonl -s pred_p025.jsonl > result_p025

/tmp2/b10902005/adl_env/bin/python test.py --top_p 0.5 --output_file pred_p05.json --model_name_or_path epoch_9 --validation_file ./data/public.json
/tmp2/b10902005/adl_env/bin/python ../data/json_to_jsonl.py --in_file pred_p05.json --out_file pred_p05.jsonl
/tmp2/b10902005/adl_env/bin/python ../ADL23-HW2/eval.py -r ../data/public.jsonl -s pred_p05.jsonl > result_p05

/tmp2/b10902005/adl_env/bin/python test.py --top_p 0.75 --output_file pred_p075.json --model_name_or_path epoch_9 --validation_file ./data/public.json
/tmp2/b10902005/adl_env/bin/python ../data/json_to_jsonl.py --in_file pred_p075.json --out_file pred_p075.jsonl
/tmp2/b10902005/adl_env/bin/python ../ADL23-HW2/eval.py -r ../data/public.jsonl -s pred_p075.jsonl > result_p075

/tmp2/b10902005/adl_env/bin/python test.py --top_k 10 --output_file pred_k10.json --model_name_or_path epoch_9 --validation_file ./data/public.json
/tmp2/b10902005/adl_env/bin/python ../data/json_to_jsonl.py --in_file pred_k10.json --out_file pred_k10.jsonl
/tmp2/b10902005/adl_env/bin/python ../ADL23-HW2/eval.py -r ../data/public.jsonl -s pred_k10.jsonl > result_k10

/tmp2/b10902005/adl_env/bin/python test.py --top_k 20 --output_file pred_k20.json --model_name_or_path epoch_9 --validation_file ./data/public.json
/tmp2/b10902005/adl_env/bin/python ../data/json_to_jsonl.py --in_file pred_k20.json --out_file pred_k20.jsonl
/tmp2/b10902005/adl_env/bin/python ../ADL23-HW2/eval.py -r ../data/public.jsonl -s pred_k20.jsonl > result_k20
