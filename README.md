# Applied Deep Learning HW2 <br>Chinese News Summarization

### Task Description

We are provided with Chinese news as input for this task, and our objective is to generate the corresponding title.

### How to Run

```bash
bash ./download.sh
bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl
```

#### `download.sh`

Download the trained model.

#### `run.sh`

Input the test data into the model and generate the predicted results. 

### Complete Training Process

##### Step 1 Turn jsonl file to json file

```bash
$ python jsonl_to_json.py --in_file ${1} --out_file tmp.json --test
$ python3 jsonl_to_json.py --in_file train.jsonl --out_file train.json
```

After completing this step, `input.jsonl` and `train.jsonl` will be separately converted to `tmp.json` and `train.json`. Since the input file lacks the 'title' column, an additional parameter is added to distinguish it.

##### Step 2 Training

```bash
$ python train.py
```

After completing this step, a fully trained model will be generated.

The code is primarily modified from the training section of `run_summarization_no_trainer.py`[1], extracting only the training part and removing the validation process.

The hyperparameters used are as follows:

| arguments                       | value               |
| ------------------------------- | ------------------- |
| `--train_file`                  | `./data/train.json` |
| `--max_source_length`           | `1024`              |
| `--max_target_length`           | `64`                |
| `--model_name_or_path`          | `google/mt5-small`  |
| `--text_column`                 | `maintext`          |
| `--summary_column`              | `title`             |
| `--per_device_train_batch_size` | `4`                 |
| `--learning_rate`               | `1e-4`              |
| `--num_train_epochs`            | `10`                |
| `--gradient_accumulation_steps` | `4`                 |
| `--output_dir`                  | `model_e10b4`       |
| `--seed`                        | `42`                |

##### Step 3 Testing

```bash
$ python test.py
```

The code is primarily modified from `run_summarization_no_trainer.py`[1], focusing only on the validation part and outputting the final predictions in `json` format.

Additionally, the following hyperparameters have been added:

1. Introduced sampling parameters, such as `top_k`, `top_p`, and `do_sampling`.
2. Added generation-controlling parameters, such as `temperature`, `length_penalty`, and `repetition`.

The hyperparameters used are as follows:

| arguments                 | value             |
| ------------------------- | ----------------- |
| `--validation_file`       | `./data/tmp.json` |
| `--output_file`           | `prediction.json` |
| `--max_source_length`     | `1024`            |
| `--val_max_target_length` | `64`              |
| `--num_beams`             | `5`               |
| `--temperature`           | `0.5`             |
| `--model_name_or_path`    | `model_e10b4`     |
| `--text_column`           | `maintext`        |
| `--summary_column`        | `title`           |

In the report, the rationale for choosing beam search over top-k or top-p sampling will be explained. 

##### Step 4 turn json file to jsonl file

```bash
$ python json_to_jsonl.py --in_file prediction.json --out_file ${2}
```

"As the final output needs to be in JSONL format, it is necessary to convert the JSON to JSONL."

Reference: 

[1] `run_summarization_no_trainer.py` source code

https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization_no_trainer.py

[2] Summarization related work

https://huggingface.co/docs/transformers/tasks/summarization

Homework Spec: 

https://docs.google.com/presentation/d/1yJEQUtzFREeuEnkBTXei4SFEftnmnP3i05m0J5aGsg8/edit#slide=id.gd486405158_0_68
