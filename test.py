from load_dataset import *

data = load_flores200_few_shot_in_context('dev', 'devtest', 'eng_Latn', 'deu_Latn', "English: ", "German: ", 3)
print(data['src'][26])
print(data['eng_Latn'][26])
print(data['deu_Latn'][26])