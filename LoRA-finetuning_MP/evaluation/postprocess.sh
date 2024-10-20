#!/bin/bash
source ~/.bashrc
conda activate /export/data2/yleung/master_thesis
python postprocess.py -s "cs" -t "en" -p "/home/yleung/LoRA-finetuning/zero_shot/cs-en_predictions.csv" -sd "/home/yleung/LoRA-finetuning/zero_shot"
python postprocess.py -s "de" -t "en" -p "/home/yleung/LoRA-finetuning/zero_shot/de-en_predictions.csv" -sd "/home/yleung/LoRA-finetuning/zero_shot"
python postprocess.py -s "gu" -t "en" -p "/home/yleung/LoRA-finetuning/zero_shot/gu-en_predictions.csv" -sd "/home/yleung/LoRA-finetuning/zero_shot"
python postprocess.py -s "is" -t "en" -p "/home/yleung/LoRA-finetuning/zero_shot/is-en_predictions.csv" -sd "/home/yleung/LoRA-finetuning/zero_shot"
python postprocess.py -s "ru" -t "en" -p "/home/yleung/LoRA-finetuning/zero_shot/ru-en_predictions.csv" -sd "/home/yleung/LoRA-finetuning/zero_shot"
python postprocess.py -s "zh" -t "en" -p "/home/yleung/LoRA-finetuning/zero_shot/zh-en_predictions.csv" -sd "/home/yleung/LoRA-finetuning/zero_shot"
python postprocess.py -s "en" -t "cs" -p "/home/yleung/LoRA-finetuning/zero_shot/en-cs_predictions.csv" -sd "/home/yleung/LoRA-finetuning/zero_shot"
python postprocess.py -s "en" -t "de" -p "/home/yleung/LoRA-finetuning/zero_shot/en-de_predictions.csv" -sd "/home/yleung/LoRA-finetuning/zero_shot"
python postprocess.py -s "en" -t "is" -p "/home/yleung/LoRA-finetuning/zero_shot/en-is_predictions.csv" -sd "/home/yleung/LoRA-finetuning/zero_shot"
python postprocess.py -s "en" -t "ru" -p "/home/yleung/LoRA-finetuning/zero_shot/en-ru_predictions.csv" -sd "/home/yleung/LoRA-finetuning/zero_shot"
python postprocess.py -s "en" -t "zh" -p "/home/yleung/LoRA-finetuning/zero_shot/en-zh_predictions.csv" -sd "/home/yleung/LoRA-finetuning/zero_shot"

