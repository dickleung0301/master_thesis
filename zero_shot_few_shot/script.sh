#!/bin/bash
source ~/.bashrc
conda activate /export/data2/yleung/master_thesis
python zs_fs.py -i "few_shot" -n 3 -s "eng_Latn" -t "deu_Latn" -m '6'
python zs_fs.py -i "few_shot" -n 3 -s "eng_Latn" -t "fra_Latn" -m '6'
python zs_fs.py -i "few_shot" -n 3 -s "eng_Latn" -t "zho_Hant" -m '6'
python zs_fs.py -i "few_shot" -n 3 -s "eng_Latn" -t "jpn_Jpan" -m '6'
python zs_fs.py -i "few_shot" -n 3 -s "eng_Latn" -t "kor_Hang" -m '6'
python zs_fs.py -i "few_shot" -n 3 -s "deu_Latn" -t "eng_Latn" -m '6'
python zs_fs.py -i "few_shot" -n 3 -s "fra_Latn" -t "eng_Latn" -m '6'
python zs_fs.py -i "few_shot" -n 3 -s "zho_Hant" -t "eng_Latn" -m '6'
python zs_fs.py -i "few_shot" -n 3 -s "jpn_Jpan" -t "eng_Latn" -m '6'
python zs_fs.py -i "few_shot" -n 3 -s "kor_Hang" -t "eng_Latn" -m '6'