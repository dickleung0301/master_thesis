import os

# setting the visible device
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

from fine_tuning import *
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_choice', type=int, help='model to be fine-tuned')
    parser.add_argument('-s', '--src_lang', type=str, help='source language of the translation direction')
    parser.add_argument('-t', '--trg_lang', type=str, help='target language of the translation direction')
    parser.add_argument('--train', dest='train', action='store_true', help='To enable fine tuning')
    parser.add_argument('--no-train', dest='train', action='store_false', help='To disable fine tuning')
    parser.add_argument('--inference', dest='inference', action='store_true', help='To enable inference')
    parser.add_argument('--no-inference', dest='inference', action='store_false', help='To disable inference')
    parser.add_argument('--mask', dest='masking', action='store_true', help='To enable masking')
    parser.add_argument('--no-mask', dest='masking', action='store_false', help='To disable masking')
    parser.add_argument('-bs', '--mini_batch_size', type=int, help='mini batch size for the fine-tuning')
    parser.add_argument('-ga', '--grad_accum', type=int, help='grad accum for the fine-tuning')
    parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate of the fine-tuning')
    parser.add_argument('-n', '--num_epochs', type=int, help='the number of epochs of the fine-tuning')
    parser.add_argument('-sd', '--save_dir', type=str, help='the saving directory for model, tokenizer and inference results')

    # get the arguments
    args = parser.parse_args()
    model_choice = args.model_choice
    src_lang = args.src_lang
    trg_lang = args.trg_lang
    train = args.train
    infer = args.inference
    masking = args.masking
    dir = src_lang + '-' + trg_lang
    mini_batch_size = args.mini_batch_size
    grad_accum = args.grad_accum
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    save_dir = args.save_dir
    
    # fine-tuning the model
    if train:
        print("####################\nfine-tuning the model\n####################")
        model,tokenizer = fine_tuning(model_choice=model_choice, src_lang=src_lang, trg_lang=trg_lang, dir=dir, mini_batch_size=mini_batch_size, 
        grad_accum=grad_accum ,learning_rate=learning_rate, num_epochs=num_epochs, masking=masking, save_dir=save_dir)

    # inference 
    if infer:
        print("####################\nstarting inference\n####################")
        inference(src_lang=src_lang, trg_lang=trg_lang, dir=dir, save_dir=save_dir)