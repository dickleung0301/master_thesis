from load_dataset import *
from model import *
from exception import *
from helper_function import *
import torch
from tqdm import tqdm
import json

# load the config file
with open('config.json', 'r') as f:
    config  = json.load(f)

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
data_index = 0
iter_sanity_check = 3

# get the pretrained model & tokenizer
model, tokenizer = model_factory(model_name)

# set the padding token of llama to the eos token of llama
if model_name == 'meta-llama/Llama-2-7b-chat-hf' or model_name == 'meta-llama/Meta-Llama-3.1-8B-Instruct' or model_name == "meta-llama/Meta-Llama-3.1-8B":
    tokenizer.pad_token = tokenizer.eos_token

# get the specific data for sanity check
sanity_check = load_flores200_sanity_check('dev', source_lang, target_lang, prefix_L1, prefix_L2, data_index)

# tokenize the specific data
tokenized_sanity_check = tokenize_data(sanity_check, source_lang, target_lang, tokenizer,
                                        truncation=True, MAX_LEN=MAX_LEN)

# Create attention mask
tokenized_sanity_check['attention_mask'] = [1 if token != tokenizer.pad_token_id else 0 for token in tokenized_sanity_check['input_ids'][0]]
attention_mask = torch.tensor([tokenized_sanity_check['attention_mask']])

# Create the dataloader
dataloader_sanity_check = create_dataloader(torch.tensor(tokenized_sanity_check['input_ids']).to(device),
                                        attention_mask.to(device),
                                        torch.tensor(tokenized_sanity_check['target_ids']),
                                        batch_size=1)

# sanity check loop
model.eval() # set the model to eval mode
loop = tqdm(dataloader_sanity_check, leave=True, desc='Evaluation') # set up the progress bar

for iter in range(iter_sanity_check):
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

        source_sentence = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        target_sentence = tokenizer.decode(target_ids[0], skip_special_tokens=True)
        translated_sentence = tokenizer.decode(translation[0], skip_special_tokens=True)
        translated_sentence = strip_llama_output(translated_sentence)

        print("The original source sentence:")
        print(source_sentence)
        print("\n")
        print("The original target sentence:")
        print(target_sentence)
        print("\n")
        print("The translation from the PLM:")
        print(translated_sentence)
        print("\n")