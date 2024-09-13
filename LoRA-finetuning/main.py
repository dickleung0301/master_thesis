from fine_tuning import *
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_choice', type=int, help='model to be fine-tuned')
    parser.add_argument('-s', '--src_lang', type=str, help='source language of the translation direction')
    parser.add_argument('-t', '--trg_lang', type=str, help='target language of the translation direction')
    parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate of the fine-tuning')
    parser.add_argument('-n', '--num_epochs', type=int, help='the number of epochs of the fine-tuning')
    parser.add_argument('-sd', '--save_dir', type=int, help='the saving directory for model, tokenizer and inference results')

    # get the arguments
    args = parser.parse_args()
    model_choice = args.model_choice
    src_lang = args.src_lang
    trg_lang = args.trg_lang
    dir = src_lang + '-' + trg_lang
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    save_dir = args.save_dir

    # check the available device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # fine-tuning the model
    print("####################\nfine-tuning the model\n####################")
    model,tokenizer = fine_tuning(model_choice=model_choice, src_lang=src_lang, trg_lang=trg_lang,
                            dir=dir, learning_rate=learning_rate, num_epochs=num_epochs, device=device, save_dir=save_dir)

    # inference 
    print("####################\nstarting inference\n####################")
    inference(src_lang=src_lang, trg_lang=trg_lang, dir=dir, model=model, tokenizer=tokenizer, device=device, save_dir=save_dir)