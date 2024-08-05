import sacrebleu
import os

# get the current and file directory
cwd = os.getcwd()
file_dir = '/zero_shot_result/'
source_lang = 'eng_Latn'
target_lang = 'deu_Latn'
trans_path = cwd + file_dir + source_lang +'2' + target_lang +'_trans.txt'
trg_path = cwd + file_dir + source_lang +'2' + target_lang +'_trg.txt'

# read the translation and target file
with open(trans_path, 'r') as f:
    trans_corpus = [f.read()]

with open(trg_path, 'r') as f:
    trg_corpus = [f.read()]

# calculate the corpus bleu score
bleu = sacrebleu.corpus_bleu(trans_corpus, [trg_corpus])
print(bleu.score)