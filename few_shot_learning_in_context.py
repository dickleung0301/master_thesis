from load_dataset import *
from model import *
from exception import *
import torch
from tqdm import tqdm
from sacrebleu.metrics import BLEU

# state the parameters
source_lang = 'eng_Latn'
target_lang = 'deu_Latn'
prefix = 'Translate from English to German: '
model_name = 't5-small'
MAX_LEN = 1024
MAX_LEN_OUTPUT = 1100
num_example = 3
num_inference = 5

# get the pretrained model & tokenizer
model, tokenizer = model_factory(model_name)

if model_name == 'meta-llama/Llama-2-7b-chat-hf':
    tokenizer.pad_token = tokenizer.eos_token

# set up the BLEU object for evaluation
bleu = BLEU()

# get the dev set of flores-200
flores200_dev = load_flores200_few_shot_in_context('dev', source_lang, target_lang, prefix, num_example, num_inference)

# tokenize the dev set
tokenized_flores200_dev = tokenize_data(flores200_dev, source_lang, target_lang, tokenizer,
                                        truncation=True, MAX_LEN=MAX_LEN)

# Create attention mask
tokenized_flores200_dev['attention_mask'] = [[1 if token != tokenizer.pad_token_id else 0 for token in x] for x in tokenized_flores200_dev['input_ids']]
attention_mask = torch.tensor(tokenized_flores200_dev['attention_mask'])


# Create the dataloader
dataloader_zero_shot = create_dataloader(torch.tensor(tokenized_flores200_dev['input_ids']),
                                        attention_mask,
                                        torch.tensor(tokenized_flores200_dev['target_ids']),
                                        batch_size=32)

# zero-shot-learning-loop
model.eval() # set the model to eval mode, as we are only doing in-context learning
loop = tqdm(dataloader_zero_shot, leave=True, desc='Evaluation') # set up the progress bar

for batch in loop:

    input_ids, attention_mask, target_ids = batch # in a shape of (batch, input, attn, target)

    with torch.no_grad():
        translation = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=MAX_LEN_OUTPUT)

    for i in range(num_inference):
        source_sentence = tokenizer.decode(input_ids[i], skip_special_tokens=True)
        target_sentence = tokenizer.decode(target_ids[i], skip_special_tokens=True)
        translated_sentence = tokenizer.decode(translation[i], skip_special_tokens=True)

        print("The original source sentence:")
        print(source_sentence)
        print("\n")
        print("The original target sentence:")
        print(target_sentence)
        print("\n")
        print("The translation from the PLM:")
        print(translated_sentence)
        print("\n")

        # set up the refs & hyp for evaluation
        # hyp = [translated_sentence]
        refs = [target_sentence]

        score = bleu.sentence_score(translated_sentence, refs)
        print(f'BLUE score: {score}')
        print("\n")