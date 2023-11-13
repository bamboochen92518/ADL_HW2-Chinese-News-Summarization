import os

a = f'python ../train.py --checkpointing_steps epoch'
os.system(a)

for i in range(10):
    a = f'python ../test.py --num_beams 5 --output_file e{i}.json --model_name_or_path ../model_e10b4/epoch_{i} --validation_file ./data/public.json'
    os.system(a)
    a = f'python ../data/json_to_jsonl.py --in_file e{i}.json --out_file e{i}.jsonl'
    os.system(a)
    a = f'python ../ADL23-HW2/eval.py -r ./data/public.jsonl -s e{i}.jsonl > result_e{i}'
    os.system(a)
