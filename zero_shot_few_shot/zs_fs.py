import argparse
from inference import *

if __name__ == "main":

    parser = argparse.ArgumentParser()
    # Argument for inference type
    parser.add_argument('-i', '--inference_type', type=str, required=True, choices=['zero_shot', 'few_shot'],
                        help='Please enter "zero_shot" or "few_shot"')

    # Argument for the number of examples for few_shot (optional, default None)
    parser.add_argument('-n', '--num_example', type=int, default=None, 
                        help='Enter the number of examples for few_shot (default is None)')

    # Source language (optional, default 'eng_Latn')
    parser.add_argument('-s', '--src_lang', type=str, default='eng_Latn', 
                        help='Source language (default: eng_Latn)')

    # Target language (optional, default 'deu_Latn')
    parser.add_argument('-t', '--trg_lang', type=str, default='deu_Latn', 
                        help='Target language (default: deu_Latn)')

    # Model choice (optional, default '6')
    parser.add_argument('-m', '--model_choice', type=str, default='6', 
                        help='Model choice (default: 6, check the config file)')

    # get the arguments
    args = parser.parse_args()
    inference_type = args.inference_type
    num_example = args.num_example
    src_lang = args.src_lang
    trg_lang = args.trg_lang
    model_choice = args.model_choice

    inference(inference_type, num_example, src_lang, trg_lang, model_choice)