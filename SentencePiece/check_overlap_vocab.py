import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer
import sentencepiece as spm

# specify the paths of the tokenizers
dir = '/export/data2/yleung'
ja = dir + '/smp_ja/ja_32000.model'
ko = dir + '/smp_ko/ko_32000.model'
zht = dir + '/smp_zht/zht_32000.model'

# load the tokenizers
sp_ja = spm.SentencePieceProcessor()
sp_ko = spm.SentencePieceProcessor()
sp_zht = spm.SentencePieceProcessor()

sp_ja.load(ja)
sp_ko.load(ko)
sp_zht.load(zht)

# Extract the vocabs from the tokenizers
sp_ja_vocab = set([sp_ja.id_to_piece(i) for i in range(sp_ja.get_piece_size())])
sp_ko_vocab = set([sp_ko.id_to_piece(i) for i in range(sp_ko.get_piece_size())])
sp_zht_vocab = set([sp_zht.id_to_piece(i) for i in range(sp_zht.get_piece_size())])

# print the informations of each tokenizer
print("##########Japanese Tokenizer##########")
print(f"Vocab size: {len(sp_ja_vocab)}")
print("Vocab of Japanese Tokenizer:")
print(sp_ja_vocab)
print("##########Korean Tokenizer##########")
print(f"Vocab size: {len(sp_ko_vocab)}")
print("Vocab of Korean Tokenizer:")
print(sp_ko_vocab)
print("##########Traditional Chinese Tokenizer##########")
print(f"Vocab size: {len(sp_zht_vocab)}")
print("Vocab of Traditional Chinese Tokenizer:")
print(sp_zht_vocab)

# load llama tokenizer
load_dotenv()
token = os.getenv('HUGGINGFACE_TOKEN')

login(token=token)

llama_3I_tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Meta-Llama-3.1-8B-Instruct', 
    cache_dir="/export/data2/yleung/model_cache", 
    token=token)

# Extract vocabs from llama tokenizer
llama_3I_vocab = set(llama_3I_tokenizer.get_vocab().keys())
print("##########Llama3.1 Instruct##########")
print(f"Vocab Size: {len(llama_3I_vocab)}")
print("Vocab of Llama3.1 Instruct Tokenizer:")
print(llama_3I_vocab)

# calculate the percentage of overlapping
overlap_vocab_ja = sp_ja_vocab.intersection(llama_3I_vocab)
overlap_vocab_ko = sp_ko_vocab.intersection(llama_3I_vocab)
overlap_vocab_zht = sp_zht_vocab.intersection(llama_3I_vocab)

overlap_percentage_ja = len(overlap_vocab_ja) / len(sp_ja_vocab) * 100
overlap_percentage_ko = len(overlap_vocab_ko) / len(sp_ko_vocab) * 100
overlap_percentage_zht = len(overlap_vocab_zht) / len(sp_zht_vocab)* 100

print("##########Overlapping vocab (given the numerator: overlapped vocab, the denominator: vocab size of the trained tokenizer)##########")
print(f"Overlap percentage of Japanese: {overlap_percentage_ja} %")
print(f"Overlap percentage of Korean: {overlap_percentage_ko} %")
print(f"Overlap percentage of Traditional Chinese: {overlap_percentage_zht} %")
