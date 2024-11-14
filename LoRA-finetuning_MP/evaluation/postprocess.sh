#!/bin/bash
source ~/.bashrc
conda activate /export/data2/yleung/master_thesis
python postprocess.py -s "en" -t "yue" -p "/home/yleung/LoRA-finetuning/adapted_result/en-yue_predictions.csv" -sd "/home/yleung/LoRA-finetuning/adapted_result"
python postprocess.py -s "en" -t "yue" -p "/home/yleung/LoRA-finetuning/evaluation/baseline/en-yue_predictions.csv" -sd "/home/yleung/LoRA-finetuning/evaluation/baseline"