# Plot the Learning Curve

### Step 1. Predict Result

```bash
$ python run.py
```

執行這步時，會先進行 training，並在每個epoch結束時存下當前的 model 。接著，對於每個 epoch 進行預測，產生`e{i}.jsonl`, `e{i}.jsonl`，最後計算分數，並存入`result_e{i}`。

### Step 2. Plot Learning Curve

```bash
$ python learning_curve.py --target rouge-1
$ python learning_curve.py --target rouge-2
$ python learning_curve.py --target rouge-l
```

