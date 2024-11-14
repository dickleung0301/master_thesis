import pandas as pd
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt

def freeze_body_lora_embedd(model):

    # freeze the transformer body
    for param in model.parameters():
        param.requires_grad = False

    for param in model.lm_head.parameters():
        param.requires_grad = True

    for param in model.model.embed_tokens.parameters():
        param.requires_grad = True

    # apply lora to the word embedds & lm_head
    lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=['embed_tokens', 'lm_head'],
            lora_dropout=0.1,
            bias='none',
            task_type='CAUSAL_LM'
    )

    print("####################\nlora config.\n####################")
    print(lora_config)
    model = get_peft_model(model, lora_config)

    return model

def check_tokenizer(tokenizer_path, corpus_path, batch_size=128):

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # to store the original line, token ids, and the decoded line
    original_line = []
    token_ids = []
    decoded_line = []

    with open(corpus_path, 'r') as file:
        for _ in range(batch_size):
            line = file.readline().strip()
            original_line.append(line)
            temp = tokenizer.encode(line)
            token_ids.append(temp)
            decoded_line.append(tokenizer.decode(temp, skip_special_tokens=True))

    output = {
        'original_line': original_line,
        'token_ids': token_ids,
        'decoded_line': decoded_line
    }
    df = pd.DataFrame(output)
    df.to_csv('check_tokenizer.csv', index=False)

def vocab_len_stat(tokenizer_path):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    vocab = tokenizer.get_vocab()

    tokens_length = [len(token) for token in vocab.keys()]

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(tokens_length, bins=range(1, max(tokens_length) + 2), edgecolor='black', align='left')
    plt.xlabel("Token Length (in Chinese characters)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Token Lengths")
    plt.xticks(range(1, max(tokens_length) + 1))

    plt.savefig("token_length_distribution.png", format="png", dpi=300)