## Quick Start

### Requiments
- pytorch
- torchvision
- tqdm
- salesforce-lavis
- sklearn
- json5
- openai

### Data Preparation

To reproduce the results in the study, we provide the pre-extracted features of the VLM in [this link](https://disk.pku.edu.cn/link/AA3641EABF29EE483F8AE89E1C149DD496) and the outputs of the LLM in [`dataset/charades-sta/llm_outputs.json`](dataset/charades-sta/llm_outputs.json) and [`dataset/activitynet/llm_outputs.json`](dataset/activitynet/llm_outputs.json). Please download the pre-extracted features and configure the path for these features in [`data_configs.py`](data_configs.py) file.

## Baseline Main Results

### Standard Split

```bash
# Charades-STA dataset
python evaluate.py --dataset charades --llm_output dataset/charades-sta/llm_outputs.json

# ActivityNet dataset
python evaluate.py --dataset activitynet --llm_output dataset/activitynet/llm_outputs.json
```

### OOD Splits

```bash
# Charades-STA OOD-1
python evaluate.py --dataset charades --split OOD-1

# Charades-STA OOD-2
python evaluate.py --dataset charades --split OOD-2

# ActivityNet OOD-1
python evaluate.py --dataset activitynet --split OOD-1

# ActivityNet OOD-2
python evaluate.py --dataset activitynet --split OOD-2
```


```bash
# Charades-CD test-ood
python evaluate.py --dataset charades --split test-ood

# Charades-CG novel-composition
python evaluate.py --dataset charades --split novel-composition

# Charades-CG novel-word
python evaluate.py --dataset charades --split novel-word
```

## Our Method Main Results

### Standard Split

```bash
# Charades-STA dataset
python evaluate_clustering.py --dataset charades --llm_output dataset/charades-sta/llm_outputs.json

# Charades-STA dataset w/ LLM
python evaluate_clustering.py --dataset charades --llm_output dataset/charades-sta/llm_outputs.json --use_llm

# ActivityNet dataset
python evaluate.py --dataset activitynet --llm_output dataset/activitynet/llm_outputs.json

# ActivityNet dataset w/ LLM
python evaluate.py --dataset activitynet --llm_output dataset/activitynet/llm_outputs.json --use_llm
```

### OOD Splits

```bash
# Charades-STA OOD-1
python evaluate_clustering.py --dataset charades --split OOD-1

# Charades-STA OOD-2
python evaluate_clustering.py --dataset charades --split OOD-2

# ActivityNet OOD-1
python evaluate_clustering.py --dataset activitynet --split OOD-1

# ActivityNet OOD-2
python evaluate_clustering.py --dataset activitynet --split OOD-2
```


```bash
# Charades-CD test-ood
python evaluate_clustering.py --dataset charades --split test-ood

# Charades-CG novel-composition
python evaluate_clustering.py --dataset charades --split novel-composition

# Charades-CG novel-word
python evaluate_clustering.py --dataset charades --split novel-word
```

### Use CLIP as VLM
```bash
# Charades-STA dataset
python evaluate_clustering.py --dataset charades_clip --llm_output dataset/charades-sta/llm_outputs.json
```