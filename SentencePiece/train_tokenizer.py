import os
import sentencepiece as spm
from sampling import *

# state the dir of the tokenizer & the path of the trained model and corpus
dir = '/export/data2/yleung/smp_ko'
corpus = '/ko.txt'
model = '/ko_32000'
sampled_corpus = '/sampled_ko.txt'

# sample the corpus 
if not os.path.exists(dir + sampled_corpus):
    sample_corpus(dir + corpus, dir + sampled_corpus)

# train the tokenizer
spm.SentencePieceTrainer.train(
    input = dir + sampled_corpus,
    model_prefix = dir + model,
    model_type='bpe',
    vocab_size = 32000,
    character_coverage=0.9995
)