import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from load_data import *
from load_model import *
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

def fine_tuning(model_choice, src_lang, trg_lang, dir, learning_rate, num_epochs, device, save_dir):

    # load model and tokenizer
    model, tokenizer = model_factory(model_choice=model_choice)

    # model information
    print("####################\nmodel info.\n####################")
    print(model)

    # applying LoRA to the model
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

    # prepare the model for training with quantisation
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.to(device)

    # load alma
    training_dataset = load_alma(split='train', dir=dir)
    eval_dataset = load_alma(split='validation', dir=dir)

    # preprocess the dataset
    processed_training_dataset = finetuning_preprocess(dataset=training_dataset, key='translation', src_lang=src_lang, trg_lang=trg_lang,
                                        tokenizer=tokenizer)
    processed_eval_dataset = finetuning_preprocess(dataset=eval_dataset, key='translation', src_lang=src_lang, trg_lang=trg_lang,
                                        tokenizer=tokenizer)

    # setting the EarlyStoppingCallback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.1
    )
    print("####################\nearly stopping config.\n####################")
    print(f"Patience: {early_stopping_callback.early_stopping_patience}")
    print(f"Threshold: {early_stopping_callback.early_stopping_threshold}")


    # training config
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_dir="./logs",
        fp16=True,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=100,
    )

    print("####################\ntraining_args config.\n####################")
    print(training_args)

    # initialise Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_training_dataset,
        eval_dataset=processed_eval_dataset,
        callbacks=[early_stopping_callback],
    )

    # fine-tune the model
    trainer.train()

    # save the pretrained model & the tokenizer
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    return model, tokenizer

def inference(src_lang, trg_lang, dir, model, tokenizer, device, save_dir):

    # load wmt dataset
    test_dataset = load_wmt22(dir=dir)

    # preprocess the dataset
    processed_test_dataset = generation_preprocess(dataset=test_dataset, key=dir, src_lang=src_lang,
                                                    tokenizer=tokenizer, device=device)


    # pack the dataset into dataloader
    test_dataloader = DataLoader(processed_test_dataset, batch_size=8, shuffle=False)

    # eval loop
    model.eval()
    predictions = []

    for batch in tqdm(test_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # inference
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=256,
                                    do_sample=False, temperature=1.0, top_p=1.0)

        # decode the outputs
        decoded_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # append the prediction
        predictions.extend(decoded_predictions)

    df = pd.DataFrame(predictions, columns=['predictions'])
    df.to_csv(save_dir + 'predictions.csv', index=False)