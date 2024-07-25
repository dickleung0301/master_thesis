from load_dataset import *
from model import *
from exception import *
import torch
from tqdm import tqdm
from torch.optim import AdamW
from sacrebleu.metrics import BLEU
import json

# load the config file
with open('config.json', 'r') as f:
    config = json.load(f)

# load the prefix and model dictionary
prefix = config['prefix']
model = config['model']

# state the parameters
source_lang = 'eng_Latn'
target_lang = 'nld_Latn'
prefix_L1 = prefix[source_lang]
prefix_L2 = prefix[target_lang]
model_choice = '1'
model_name = model[model_choice]
MAX_LEN = 128
MAX_LEN_OUTPUT = 200
count = 0
iter_for_showing_result = 10
num_samples = 5
num_epochs = 5
model_path = './few_shot_learned_model'
save_model = False

# get the pretrained model & tokenizer
model, tokenizer = model_factory(model_name)

# set up the BLEU object for evaluation
bleu = BLEU()

if model_name == 'meta-llama/Llama-2-7b-chat-hf' or model_name == 'meta-llama/Meta-Llama-3.1-8B-Instruct' or model_name == "meta-llama/Meta-Llama-3.1-8B":
    tokenizer.pad_token = tokenizer.eos_token

# get samples from dev set for few-shot learning
flores200_few_shot = load_flores200_few_shot('dev', source_lang, target_lang, prefix_L1, prefix_L2, num_samples)

# get the devtest set for evaluation
flores200_devtest = load_flores200('devtest', source_lang, target_lang, prefix_L1, prefix_L2)

# tokenize the dev & devtest set
tokenized_flores200_few_shot = tokenize_data(flores200_few_shot, source_lang, target_lang, tokenizer,
                                            truncation=True, MAX_LEN=MAX_LEN)

tokenized_flores200_devtest = tokenize_data(flores200_devtest, source_lang, target_lang, tokenizer,
                                            truncation=True, MAX_LEN=MAX_LEN)

# create attention mask
tokenized_flores200_few_shot['attention_mask'] = [[1 if token != tokenizer.pad_token_id else 0 for token in x] for x in tokenized_flores200_few_shot['input_ids']]
attention_mask_few_shot = torch.tensor(tokenized_flores200_few_shot['attention_mask'])
tokenized_flores200_devtest['attention_mask'] = [[1 if token != tokenizer.pad_token_id else 0 for token in x] for x in tokenized_flores200_devtest['input_ids']]
attention_mask_devtest = torch.tensor(tokenized_flores200_devtest['attention_mask'])

# create dataloader
dataloader_few_shot = create_dataloader(torch.tensor(tokenized_flores200_few_shot['input_ids']),
                                        attention_mask_few_shot,
                                        torch.tensor(tokenized_flores200_few_shot['target_ids']),
                                        batch_size=num_samples)
dataloader_devtest = create_dataloader(torch.tensor(tokenized_flores200_devtest['input_ids']),
                                        attention_mask_devtest,
                                        torch.tensor(tokenized_flores200_devtest['target_ids']),
                                        batch_size=32)

# training-loop for few-shot learning
model.train()
optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(num_epochs):

    # set up the progress bar
    loop = tqdm(dataloader_few_shot, leave=True, desc=f'Epoch {epoch + 1}')

    for batch in loop:
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
        loop.set_postfix(loss=loss.item())

# evaluation of the learned PLM
model.eval()
loop = tqdm(dataloader_devtest, leave=True, desc='Evaluation')

for batch in loop:
    input_ids, attention_mask, target_ids = batch

    with torch.no_grad():
        translation = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=MAX_LEN_OUTPUT)

    # print the first pair of translation in every 10 batches
    if count % iter_for_showing_result == 0:

        source_sentence = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        target_sentence = tokenizer.decode(target_ids[0], skip_special_tokens=True)
        translated_sentence = tokenizer.decode(translation[0], skip_special_tokens=True)

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

    count += 1

# save the few-shot learned model
if save_model:
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)