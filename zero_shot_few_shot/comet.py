import json
from zero_shot_few_shot.comet import download_model, load_from_checkpoint

file_dir = './zero_shot_result/'
source_lang = 'eng_Latn'
target_lang = 'gle_Latn'

data_path = file_dir + source_lang + '2' + target_lang + '_comet'
save_path = file_dir + source_lang + '2' + target_lang + '.comet'

model_path = download_model("Unbabel/wmt22-comet-da")

model = load_from_checkpoint(model_path)

data_list = []

with open(data_path, 'r') as file:
    for line in file:
      line = line.strip()
      if line:
        data = json.loads(line)
        data_list.append(data)

model_output = model.predict(data_list, batch_size=32, gpus=1)
print(model_output.system_score)

with open(save_path, 'w') as f:
  f.write(f'COMET Score: {model_output.system_score}\n')