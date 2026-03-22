import torch
import os
import pandas as pd


from MLLM_models import *
from v1_expalanation_utils import *
from tqdm.contrib import tzip


jbv_path = '/path/to/your/jailbreakv'

model_name = 'llava'
attack_success = torch.load(f'{jbv_path}/JailBreakV_28k/{model_name}/attack_success.pt')
model = LLaVA()
model.eval()

test_exp = init_exp(model_name, model)

query_df = pd.read_csv(f'{jbv_path}/JailBreakV_28K/JailBreakV_28K.csv')
batch_query_text = query_df["jailbreak_query"]
batch_image_path = [os.path.join(jbv_path, path) for path in
                    query_df["image_path"]]
indices = list(range(len(batch_query_text)))
random.shuffle(indices)
print("Image loaded.")
df_save = query_df
batch_response = [None] * len(batch_image_path)

attack_success = torch.load(f'{jbv_path}/results/JailBreakV_28k/{model_name}/attack_success.pt')
# try:
#     forward_info = torch.load(f'/storage/gyy/mllm/LLM_EXPLANATION/new_forward_info/forward_info_JBV_{model_name}_temp.pt')
# except Exception as e:
#     print('e is ',e)
#     forward_info = torch.load(f'/storage/gyy/mllm/LLM_EXPLANATION/new_forward_info/forward_info_JBV_{model_name}.pt')
# test_exp.forward_info = forward_info
# print(len(forward_info))
# raise
with torch.no_grad():
    # for index, (image_path, prompt) in enumerate(tzip(rand_batch_image_path, rand_batch_query_text)):
    for index in tqdm.tqdm(attack_success[len(test_exp.forward_info):]):
    
        image_path = batch_image_path[index]
        prompt = batch_query_text[index]
        results = model.inference(image_path, prompt, generate=False)
        test_exp.get_forward_info(results, 0, pos = -1)
        torch.save(test_exp.forward_info, f'hiddenstates/forward_info_JBV_{model_name}_temp.pt')
        torch.save(test_exp.forward_info, f'hiddenstates/forward_info_JBV_{model_name}.pt')
        # raise