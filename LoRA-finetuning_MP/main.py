import os

# setting the visible device
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ["HF_DATASETS_CACHE"] = "/export/data2/yleung/dataset"

from fine_tuning import *
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_choice', type=int, help='model to be fine-tuned')
    parser.add_argument('-s', '--src_lang', type=str, default=None, help='source language of the translation direction')
    parser.add_argument('-t', '--trg_lang', type=str, default=None, help='target language of the translation direction')
    parser.add_argument('--train', dest='train', action='store_true', help='To enable fine tuning')
    parser.add_argument('--vocab_adapt', dest='vocab_adapt', action='store_true', help='To carry out vocabulary adaptation')
    parser.add_argument('--lora', dest='lora', action='store_true', help='To apply lora when carrying out vocabulary adaptation')
    parser.add_argument('--sanity_check', dest='sanity_check', action='store_true', help='To carry out sanity check for different settings')
    parser.add_argument('--mono_train', dest='mono_train', action='store_true', help='To carry out monolingual corpus training')
    parser.add_argument('--inference', dest='inference', action='store_true', help='To enable inference')
    parser.add_argument('--mask', dest='masking', action='store_true', help='To enable masking')
    parser.add_argument('--right', dest='right_padding', action='store_true', help='To enable right padding in inference')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='To run inference with baseline')
    parser.add_argument('-bs', '--mini_batch_size', type=int, help='mini batch size for the fine-tuning')
    parser.add_argument('-ga', '--grad_accum', type=int, help='grad accum for the fine-tuning')
    parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate of the fine-tuning')
    parser.add_argument('-n', '--num_epochs', type=int, help='the number of epochs of the fine-tuning')
    parser.add_argument('--wmt22', dest='wmt22', action='store_true', help='To use WMT22 dataset for inference')
    parser.add_argument('--wmt19', dest='wmt19', action='store_true', help='To use WMT19 dataset for inference')
    parser.add_argument('-sd', '--save_dir', type=str, help='the saving directory for model, tokenizer and inference results')
    parser.add_argument('--mono_corpus_train', type=str, help='the path of the training monolingual corpus')
    parser.add_argument('--mono_corpus_eval', type=str, help='the path of the eval monolingual corpus')
    parser.add_argument('--tokenizer_path', type=str, help='the path of the trained sentencepiece tokenizer')
    parser.add_argument('--train_num_line', type=int, help='the number of lines for training')
    parser.add_argument('--eval_num_line', type=int, help='the number of line for evaluation')

    # get the arguments
    args = parser.parse_args()
    model_choice = args.model_choice
    src_lang = args.src_lang
    trg_lang = args.trg_lang
    train = args.train
    vocab_adapt = args.vocab_adapt
    lora = args.lora
    sanity_check = args.sanity_check
    mono_train = args.mono_train
    infer = args.inference
    masking = args.masking
    right_padding = args.right_padding
    baseline = args.baseline
    wmt22 = args.wmt22
    wmt19 = args.wmt19
    if src_lang != None and trg_lang != None:
        dir = src_lang + '-' + trg_lang
    else:
        dir = None
    mini_batch_size = args.mini_batch_size
    grad_accum = args.grad_accum
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    save_dir = args.save_dir
    mono_corpus_train = args.mono_corpus_train
    mono_corpus_eval = args.mono_corpus_eval
    tokenizer_path = args.tokenizer_path
    train_num_line = args.train_num_line
    eval_num_line = args.eval_num_line
    
    # fine-tuning the model
    if train:
        print("####################\nfine-tuning the model\n####################")
        model,tokenizer = fine_tuning(model_choice=model_choice, vocab_adapt=vocab_adapt, lora=lora, sanity_check=sanity_check,
        mono_train=mono_train, mono_corpus_train=mono_corpus_train, mono_corpus_eval=mono_corpus_eval, tokenizer_path=tokenizer_path,
        src_lang=src_lang, trg_lang=trg_lang, dir=dir, mini_batch_size=mini_batch_size, grad_accum=grad_accum, learning_rate=learning_rate,
        num_epochs=num_epochs, masking=masking, save_dir=save_dir, train_num_line=train_num_line, eval_num_line=eval_num_line)

    # inference 
    if infer:
        print("####################\nstarting inference\n####################")
        inference(src_lang=src_lang, trg_lang=trg_lang, dir=dir, save_dir=save_dir, right_padding=right_padding,
        baseline=baseline, model_choice=model_choice, wmt22=wmt22, wmt19=wmt19)