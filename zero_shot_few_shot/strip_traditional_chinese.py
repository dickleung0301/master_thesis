file_path = '/Users/yiuchungleung/Desktop/Masterarbeit/baseline_model/few_shot_in_context_presentation/eng_Latn2zho_Hant_trans.txt'
save_path = '/Users/yiuchungleung/Desktop/Masterarbeit/baseline_model/few_shot_in_context_presentation/cleaned_eng_Latn2zho_Hant_trans.txt'

with open(file_path, 'r') as infile, open(save_path, 'w') as outfile:
    for line in infile:
        striped_line = line.split('：')[-1]
        outfile.write(striped_line)