# Vista-LLaMA


### Vista-LLaMA: educing Hallucination in Video Language Models via Equal Distance to Visual Tokens




## Installation :wrench:

We recommend setting up a conda environment for the project:
```shell
bash script/set_env.sh
```
---

## Dataset

Follow the instructions in [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT)
If Eva-CLIP-G is used to extract video features, use the scripts/save_evaclip_feature to extract activitynet features

## Training Inference and Evalaution:

```shell
bash script/train_val_eval.sh
```


## Test model on the Next-QA dataset

```shell
bash script/run_infer.sh
```
