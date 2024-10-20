#!/bin/bash
source ~/.bashrc
conda activate /export/data2/yleung/master_thesis
python main.py -m 2 -s "gu" -t "en" --inference --wmt19 -sd "/export/data2/yleung/full_alma"
python main.py -m 2 -s "cs" -t "en" --inference --baseline --wmt22 -sd "/export/data2/yleung/baseline"
python main.py -m 2 -s "de" -t "en" --inference --baseline --wmt22 -sd "/export/data2/yleung/baseline"
python main.py -m 2 -s "is" -t "en" --inference --baseline --wmt22 -sd "/export/data2/yleung/baseline"
python main.py -m 2 -s "ru" -t "en" --inference --baseline --wmt22 -sd "/export/data2/yleung/baseline"
python main.py -m 2 -s "zh" -t "en" --inference --baseline --wmt22 -sd "/export/data2/yleung/baseline"
python main.py -m 2 -s "en" -t "cs" --inference --baseline --wmt22 -sd "/export/data2/yleung/baseline"
python main.py -m 2 -s "en" -t "de" --inference --baseline --wmt22 -sd "/export/data2/yleung/baseline"
python main.py -m 2 -s "en" -t "is" --inference --baseline --wmt22 -sd "/export/data2/yleung/baseline"
python main.py -m 2 -s "en" -t "ru" --inference --baseline --wmt22 -sd "/export/data2/yleung/baseline"
python main.py -m 2 -s "en" -t "zh" --inference --baseline --wmt22 -sd "/export/data2/yleung/baseline"
python main.py -m 2 -s "gu" -t "en" --inference --baseline --wmt19 -sd "/export/data2/yleung/baseline"