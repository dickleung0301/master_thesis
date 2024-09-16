from zero_shot_few_shot.load_dataset import *
from zero_shot_few_shot.model import *
from zero_shot_few_shot.exception import *
from zero_shot_few_shot.helper_function import *
import torch
from tqdm import tqdm
from torch.optim import AdamW
import json
import os
from peft import prepare_model_for_kbit_training

# load the config file
with open('config.json', 'r') as f:
    config = json.load(f)

# load the prefix and model dictionary
prefix = config['prefix']
model = config['model']

# check the device on the machine
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# state the parameters
source_lang = 'eng_Latn'
target_lang = 'deu_Latn'
prefix_L1 = prefix[source_lang]
prefix_L2 = prefix[target_lang]
model_choice = '6'
model_name = model[model_choice]
MAX_LEN = 128
MAX_LEN_OUTPUT = 128
num_samples = 4
model_path = './few_shot_learned_model'
save_model = False

# get the current working directory
cwd = os.getcwd()
save_directory = cwd + '/few_shot_result'

# the corpus for translation and target sentence
translations = ''
target_sentences = ''

# get the pretrained model & tokenizer
model, tokenizer = model_factory(model_name)

if model_name == 'meta-llama/Llama-2-7b-chat-hf' or model_name == 'meta-llama/Meta-Llama-3.1-8B-Instruct' or model_name == "meta-llama/Meta-Llama-3.1-8B":
    tokenizer.pad_token = tokenizer.bos_token
    tokenizer.padding_side = 'left'

# get samples from devtest set for few-shot learning
flores200_few_shot = load_flores200_few_shot('devtest', source_lang, target_lang, prefix_L1, prefix_L2, num_samples)

# get the dev set for evaluation
flores200_dev = load_flores200('dev', source_lang, target_lang, prefix_L1, prefix_L2)

# tokenize the dev & devtest set
tokenized_flores200_few_shot = tokenize_data(flores200_few_shot, source_lang, target_lang, tokenizer,
                                            truncation=True, MAX_LEN=MAX_LEN)

tokenized_flores200_dev = tokenize_data(flores200_dev, source_lang, target_lang, tokenizer,
                                            truncation=True, MAX_LEN=MAX_LEN)

# create attention mask
tokenized_flores200_few_shot['attention_mask'] = [[1 if token != tokenizer.pad_token_id else 0 for token in x] for x in tokenized_flores200_few_shot['input_ids']]
attention_mask_few_shot = torch.tensor(tokenized_flores200_few_shot['attention_mask'])
tokenized_flores200_dev['attention_mask'] = [[1 if token != tokenizer.pad_token_id else 0 for token in x] for x in tokenized_flores200_dev['input_ids']]
attention_mask_devtest = torch.tensor(tokenized_flores200_dev['attention_mask'])

# create dataloader
dataloader_few_shot = create_dataloader(torch.tensor(tokenized_flores200_few_shot['input_ids']).to(device),
                                        attention_mask_few_shot.to(device),
                                        torch.tensor(tokenized_flores200_few_shot['target_ids']).to(device),
                                        batch_size=num_samples)
dataloader_dev = create_dataloader(torch.tensor(tokenized_flores200_dev['input_ids']).to(device),
                                        attention_mask_devtest.to(device),
                                        torch.tensor(tokenized_flores200_dev['target_ids']),
                                        batch_size=32)

# training-loop for few-shot learning
model = prepare_model_for_kbit_training(model)
model.train()
optimizer = AdamW(model.parameters(), lr=5e-5)

# set up the progress bar
learning_loop = tqdm(dataloader_few_shot, leave=True, desc=f'{num_samples}_shot_learning')

for batch in learning_loop:
    input_ids, attention_mask, target_ids = batch # in a shape of (batch, input, attn, output)

    # forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
    loss = outputs.loss

    # Backprop
    loss.backward()

    # weight update
    optimizer.step()
    optimizer.zero_grad() # reset the gradient

    # print the loss
    learning_loop.set_postfix(loss=loss.item())

# evaluation of the learned PLM
model.eval()
eval_loop = tqdm(dataloader_dev, leave=True, desc='Evaluation')

for batch in eval_loop:
    input_ids, attention_mask, target_ids = batch

    with torch.no_grad():
        if model_name == 'meta-llama/Llama-2-7b-chat-hf' or model_name == 'meta-llama/Meta-Llama-3.1-8B-Instruct' or model_name == "meta-llama/Meta-Llama-3.1-8B":
            translation = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=MAX_LEN_OUTPUT,
                                         do_sample=False, temperature=1.0, top_p=1.0)
        else:
            translation = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=MAX_LEN_OUTPUT,
                                         do_sample=False, temperature=1.0, top_p=1.0)

    translation = translation.to('cpu')

    # add the translations and target sentences to the
    for trans_sent in translation:
        translated_sentence = tokenizer.decode(trans_sent, skip_special_tokens=True)
        translated_sentence = strip_llama_output(translated_sentence)
        translations += (translated_sentence + '\n')

    for trg_sent in target_ids:
        target_sentence = tokenizer.decode(trg_sent, skip_special_tokens=True)
        target_sentence = target_sentence.replace('"', '')
        target_sentences += (target_sentence + '\n')

save_corpus(translations, save_directory, source_lang, target_lang)
save_corpus(target_sentences, save_directory, source_lang, target_lang, translation=False)

# save the few-shot learned model
if save_model:
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)