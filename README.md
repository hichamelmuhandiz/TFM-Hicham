# TFM  - Exploring rejection strategies for zero-shot classification
This repository contains the code used in the TFM "Exploring rejection strategies for zero-shot classification". Particularly, it is a modified version of the CLIP_benchmark repository provided in `https://github.com/LAION-AI/CLIP_benchmark/` which includes different rejection strategies implemented to evaluate how the accuracy improves when we discard samples which the model is not confident about.
 
## How to use?
```
To install run
cd CLIP_benchmark
python setup.py install
```

To evaluate we recommend to create a models.txt like
```
ViT-B-32,openai
```

to get the list of datasets 
```
wget https://raw.githubusercontent.com/LAION-AI/CLIP_benchmark/main/benchmark/webdatasets.txt
```

Then to run

```
clip_benchmark eval --pretrained_model models.txt \
    --dataset "webdatasets.txt" \
    --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
    --output "benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json"
```
