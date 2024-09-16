from zero_shot_few_shot.load_dataset import *
from zero_shot_few_shot.model import *
from zero_shot_few_shot.exception import *
from zero_shot_few_shot.helper_function import *
import torch
from tqdm import tqdm
import json
import os

# load the config file
with open('config.json', 'r') as f:
    config = json.load(f)

# load the prefix and model dictionary
prefix = config['prefix']
model = config['model']

# define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# state the parameters
source_lang = 'eng_Latn'
target_lang = 'deu_Latn'
prefix_L1 = prefix[source_lang]
prefix_L2 = prefix[target_lang]
model_choice = '6'
model_name = model[model_choice]
MAX_LEN = 512
MAX_LEN_OUTPUT = 128
num_example = 4

# get the current working directory
cwd = os.getcwd()
save_directory = cwd + '/few_shot_in_context_result'

# the corpus for translation and target sentence
translations = ''
target_sentences = ''

# get the pretrained model & tokenizer
model, tokenizer = model_factory(model_name)

if model_name == 'meta-llama/Llama-2-7b-chat-hf' or model_name == 'meta-llama/Meta-Llama-3.1-8B-Instruct' or model_name == "meta-llama/Meta-Llama-3.1-8B":
    tokenizer.pad_token = tokenizer.bos_token
    tokenizer.padding_side = 'left'

# get the flores-200 dataset for few-shot-in-context
flores200 = load_flores200_few_shot_in_context('dev', 'devtest', source_lang, target_lang, prefix_L1, prefix_L2, num_example)

# tokenize the dev set
tokenized_flores200 = tokenize_data(flores200, source_lang, target_lang, tokenizer,
                                    truncation=True, MAX_LEN=MAX_LEN)

# Create attention mask
tokenized_flores200['attention_mask'] = [[1 if token != tokenizer.pad_token_id else 0 for token in x] for x in tokenized_flores200['input_ids']]
attention_mask = torch.tensor(tokenized_flores200['attention_mask'])


# Create the dataloader
dataloader_few_shot_in_context = create_dataloader(torch.tensor(tokenized_flores200['input_ids']).to(device),
                                        attention_mask.to(device),
                                        torch.tensor(tokenized_flores200['target_ids']),
                                        batch_size=32)

# zero-shot-learning-loop
model.eval() # set the model to eval mode, as we are only doing in-context learning
loop = tqdm(dataloader_few_shot_in_context, leave=True, desc='Evaluation') # set up the progress bar

for batch in loop:

    input_ids, attention_mask, target_ids = batch # in a shape of (batch, input, attn, target)

    with torch.no_grad():
        if model_name == 'meta-llama/Llama-2-7b-chat-hf' or model_name == 'meta-llama/Meta-Llama-3.1-8B-Instruct' or model_name == "meta-llama/Meta-Llama-3.1-8B":
            translation = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=MAX_LEN_OUTPUT,
                                         do_sample=False, temperature=1.0, top_p=1.0)
        else:
            translation = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=MAX_LEN_OUTPUT,
                                         do_sample=False, temperature=1.0, top_p=1.0)

    translation = translation.to('cpu')

    # add the translations and target sentences to the corpus
    for trans_sent in translation:
        translated_sentence = tokenizer.decode(trans_sent, skip_special_tokens=True)
        translated_sentence = strip_in_context(translated_sentence, prefix_L2)
        translations += (translated_sentence + '\n')

    for trg_sent in target_ids:
        target_sentence = tokenizer.decode(trg_sent, skip_special_tokens=True)
        target_sentences += (target_sentence + '\n')

save_corpus(translations, save_directory, source_lang, target_lang)
save_corpus(target_sentences ,save_directory, source_lang, target_lang, translation=False)