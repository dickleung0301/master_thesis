#!/bin/bash
source ~/.bashrc
conda activate /export/data2/yleung/master_thesis
python main.py -m 3 -s "zh" -t "en" --inference --baseline --wmt22 -sd "/export/data2/yleung/baseline"
python main.py -m 3 -s "en" -t "cs" --inference --baseline --wmt22 -sd "/export/data2/yleung/baseline"
python main.py -m 3 -s "en" -t "de" --inference --baseline --wmt22 -sd "/export/data2/yleung/baseline"
python main.py -m 3 -s "en" -t "is" --inference --baseline --wmt22 -sd "/export/data2/yleung/baseline"
python main.py -m 3 -s "en" -t "ru" --inference --baseline --wmt22 -sd "/export/data2/yleung/baseline"
python main.py -m 3 -s "en" -t "zh" --inference --baseline --wmt22 -sd "/export/data2/yleung/baseline"