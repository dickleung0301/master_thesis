import random

def sample_corpus(input_file, output_file, sample_ratio=0.1, seed=42):

    # state the seed of random
    random.seed(seed)

    with open(input_file, 'r', encoding='utf-8') as fin, \
        open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            if random.random() < sample_ratio:
                fout.write(line)