import os

# setting the visible device
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ["HF_DATASETS_CACHE"] = "/export/data2/yleung/dataset"

import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader
from load_data import *
from load_model import *
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator, infer_auto_device_map, dispatch_model
from accelerate.utils import set_seed
from vocab_adapt_utils import * 
from utils import *

def fine_tuning(model_choice, vocab_adapt, lora, sanity_check, mono_train, mono_corpus_train, mono_corpus_eval, tokenizer_path, src_lang, trg_lang, dir, mini_batch_size, grad_accum, learning_rate, num_epochs, masking, save_dir, train_num_line, eval_num_line):

    accelerator = Accelerator(gradient_accumulation_steps=grad_accum, mixed_precision='fp16')
    set_seed(42)

    # load model and tokenizer
    model, tokenizer = model_factory(model_choice=model_choice)
    if vocab_adapt:
        model, tokenizer = vocab_adaptation(model=model, original_tokenizer=tokenizer, tokenizer_path=tokenizer_path, lora=lora)

    # model information
    print("####################\nmodel info.\n####################")
    print(model)

    # applying LoRA to the model
    if not vocab_adapt and not sanity_check:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=['q_proj', 'v_proj'],
            lora_dropout=0.1,
            bias='none',
            task_type='CAUSAL_LM'
        )

        print("####################\nlora config.\n####################")
        print(lora_config)

        # prepare the model for training with LoRA
        model = get_peft_model(model, lora_config)
    elif sanity_check:
        model = freeze_body_lora_embedd(model)        

    max_memory = {
        0: "8GiB",  # cuda:0 -> physical GPU 0
        1: "24GiB",  # cuda:1 -> physical GPU 1
        # 2: "24GiB",  # cuda:2 -> physical GPU 3
        "cpu": "30GiB"
    }

    # Define the modules that should not be split
    no_split_modules = ["LlamaDecoderLayer","LoraLayer"]  # Ensure LoRA layers are not split

    # Infer device map
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=no_split_modules,  # Prevent splitting LoRA layers
        dtype=torch.float16
    )

    # Distribute the model
    model = dispatch_model(model, device_map=device_map)

    # load alma
    if src_lang != None and trg_lang != None and dir != None and mono_train == False:
        training_dataset = load_alma(split='train', dir=dir)
        eval_dataset = load_alma(split='validation', dir=dir)
    elif mono_train == True:
        processed_training_dataset = LineByLineDataset(mono_corpus_train, tokenizer=tokenizer, model_choice=model_choice, num_line=train_num_line)
        processed_eval_dataset = LineByLineDataset(mono_corpus_eval, tokenizer=tokenizer, model_choice=model_choice, num_line=eval_num_line)
    else:
        training_dataset = None
        eval_dataset = None

    # preprocess the dataset
    if mono_train == False:
        processed_training_dataset = finetuning_preprocess(model_choice=model_choice, dataset=training_dataset, key='translation', src_lang=src_lang, trg_lang=trg_lang,
                                                        trans_dir=dir, tokenizer=tokenizer, masking=masking)
        processed_eval_dataset = finetuning_preprocess(model_choice=model_choice, dataset=eval_dataset, key='translation', src_lang=src_lang, trg_lang=trg_lang,
                                                        trans_dir=dir, tokenizer=tokenizer, masking=masking, split='validation')

    # fit the data into a dataloader
    train_loader = DataLoader(
        processed_training_dataset,
        batch_size=mini_batch_size,
        shuffle=not mono_train,
        collate_fn=None
    )

    eval_loader = DataLoader(
        processed_eval_dataset,
        batch_size=mini_batch_size,
        shuffle=False,
        collate_fn=None
    )

    # optimizer 
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # prepare all the things with accelerator
    model, optimizer, train_loader, eval_loader = accelerator.prepare(model, optimizer, train_loader, eval_loader)

    # setting the EarlyStoppingCallback
    best_eval_loss = float('inf')
    patience = 3  # Number of epochs to wait for improvement
    early_stop_threshold = 0.01
    early_stop = False
    num_bad_steps = 0
    if not vocab_adapt:
        eff_batch_to_eval = 20
    else:
        eff_batch_to_eval = 100
    print("####################\nearly stopping config.\n####################")
    print(f"Patience: {patience}")
    print(f"Threshold: {early_stop_threshold}")

    print("####################\ntraining_args config.\n####################")
    print(f"Mini Batch Size: {mini_batch_size}")
    print(f"Gradient Accumulation: {grad_accum}")
    print(f"Effective Batch Size: {mini_batch_size * grad_accum}")
    print(f"LR: {learning_rate}")
    print(f"# Epochs:{num_epochs}")
    print(f"After {eff_batch_to_eval} Effective Batch, The Model Will Be Evaluated")

    # training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not accelerator.is_local_main_process)
        
        for step, batch in enumerate(progress_bar):
            # No need to move data; Accelerate handles it
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            # logging allocated & reserved memory for devices
            #if step % 100 == 0:
            #    print("####################\nmemory allocation\n####################")
            #    for i in range(torch.cuda.device_count()):
            #        allocated_memory = torch.cuda.memory_allocated(i) / (1024 ** 3)
            #        reserved_memory = torch.cuda.memory_reserved(i) / (1024 ** 3)
            #        print(f"GPU {i}: Allocated Memory: {allocated_memory:.2f} GB, Reserved Memory: {reserved_memory:.2f} GB")

            if (step + 1) % (grad_accum * eff_batch_to_eval) == 0:
                # Evaluate the model on the validation dataset
                model.eval()
                eval_loss = 0
                for batch in eval_loader:
                    with torch.no_grad():
                        outputs = model(**batch)
                        loss = outputs.loss
                        eval_loss += loss.item()
                avg_eval_loss = eval_loss / len(eval_loader)
                print(f"Effective Batch {(step + 1) / (grad_accum)} Validation Loss: {avg_eval_loss}")
                model.train()
                
                # Early Stopping Check
                if avg_eval_loss < best_eval_loss - early_stop_threshold:  # Threshold for improvement
                    best_eval_loss = avg_eval_loss
                    num_bad_steps = 0

                    # save the pretrained model & the tokenizer
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(save_dir, save_function=accelerator.save)
                        tokenizer.save_pretrained(save_dir)

                else:
                    num_bad_steps += 1
                    if num_bad_steps >= patience:
                        print("Early stopping triggered.")
                        early_stop = True
                        break 

        if early_stop:
            break
        
        # Calculate average training loss for the epoch
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss}")

    return model, tokenizer

def inference(src_lang, trg_lang, dir, save_dir, right_padding, baseline, model_choice, wmt22, wmt19):

    # load the access token from .env
    load_dotenv()
    token = os.getenv('HUGGINGFACE_TOKEN')

    # login huggingface_hub
    login(token=token)

    #max_memory = {
    #    0: "24GiB",  # cuda:0 -> physical GPU 0
    #    1: "24GiB",  # cuda:1 -> physical GPU 1
    #    2: "24GiB",  # cuda:2 -> physical GPU 3
    #    "cpu": "30GiB"
    #}

    # load the model from the save directory 
    if not baseline:
        model = AutoModelForCausalLM.from_pretrained(save_dir, device_map='auto', token=token)
        tokenizer = AutoTokenizer.from_pretrained(save_dir, token=token)
    else:
        model, tokenizer = model_factory(model_choice=model_choice, device_map='auto')

    # get the device of the embedding layer
    first_device = next(model.parameters()).device

    # load wmt dataset
    if wmt22:
        test_dataset = load_wmt22(dir=dir)
        # preprocess the dataset
        processed_test_dataset = generation_preprocess(model_choice=model_choice, dataset=test_dataset, key=dir, src_lang=src_lang, trg_lang=trg_lang,
                                                    trans_dir=dir, tokenizer=tokenizer, right_padding=right_padding)
    if wmt19:
        test_dataset = load_wmt19(dir=dir)
        # preprocess the dataset
        processed_test_dataset = generation_preprocess(model_choice=model_choice, dataset=test_dataset, key='translation', src_lang=src_lang, trg_lang=trg_lang,
                                                    trans_dir=dir, tokenizer=tokenizer, right_padding=right_padding)
    if src_lang == 'yue' or trg_lang == 'yue':
        test_dataset = load_yue_trans()
        # preprocess the dataset
        processed_test_dataset = generation_preprocess(model_choice=model_choice, dataset=test_dataset, src_lang=src_lang, trg_lang=trg_lang,
                                                    trans_dir=dir, tokenizer=tokenizer, right_padding=right_padding)   

    # pack the dataset into dataloader
    test_dataloader = DataLoader(processed_test_dataset, batch_size=8, shuffle=False)

    # eval loop
    model.eval()
    inputs_list = []
    labels_list = []
    predictions_list = []

    for batch in tqdm(test_dataloader):
        # move the inputs to gpu 0 as model parallelism
        input_ids = batch['input_ids'].to(first_device)
        attention_mask = batch['attention_mask'].to(first_device)
        labels = batch['labels']

        # inference
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=256,
                                     do_sample=False, temperature=1.0, top_p=1.0)

        # Move tensors to CPU for decoding
        outputs = outputs.cpu()
        input_ids = input_ids.cpu()
        labels = labels.cpu()

        # decode the outputs
        decoded_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # append the inputs, labels & predictions
        inputs_list.extend(decoded_inputs)
        labels_list.extend(decoded_labels)
        predictions_list.extend(decoded_predictions)

    # Materialisation
    materialisation = {
        'inputs': inputs_list,
        'labels': labels_list,
        'predictions': predictions_list
    }
    df = pd.DataFrame(materialisation)
    df.to_csv(save_dir + f'/{dir}_predictions.csv', index=False)