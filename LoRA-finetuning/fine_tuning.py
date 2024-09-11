import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from load_data import *
from load_model import *
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

def fine_tuning():
    
    # translation direction, src_lang, target_lang
    dir = 'zh-en'
    src_lang = 'zh'
    trg_lang = 'en'
    
    # hyper-parameter
    lr = 5e-5
    num_epochs = 3

    # check the available device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model and tokenizer
    model, tokenizer = model_factory()

    # applying LoRA to the model
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM'
    )

    model = get_peft_model(model, lora_config)
    model.to(device)

    # load alma & wmt22 dataset
    training_dataset = load_alma(split='train', dir=dir)
    eval_dataset = load_alma(split='validation', dir=dir)
    test_dataset = load_wmt22(dir=dir)

    # preprocess the dataset
    processed_training_dataset = finetuning_preprocess(dataset=training_dataset, key='translation', src_lang=src_lang, trg_lang=trg_lang,
                                        tokenizer=tokenizer)
    processed_eval_dataset = finetuning_preprocess(dataset=eval_dataset, key='translation', src_lang=src_lang, trg_lang=trg_lang,
                                        tokenizer=tokenizer)
    processed_test_dataset = generation_preprocess(dataset=test_dataset, key=dir, src_lang=src_lang, trg_lang=trg_lang,
                                        tokenizer=tokenizer)

    # setting the EarlyStoppingCallback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.01
    )

    # training config
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        logging_dir="./logs",
        fp16=False,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=100,
    )

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
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=128,
                                    do_sample=False, temperature=1.0, top_p=1.0, repetition_penalty=1.2)

        # decode the outputs
        decoded_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # append the prediction
        predictions.extend(decoded_predictions)

    # print out the predictions
    for i, prediction in enumerate(predictions):
        print(f"Prediction {i + 1}: {prediction}")

    df = pd.DataFrame(predictions, columns=['predictions'])
    df.to_csv('./predictions.csv', index=False)