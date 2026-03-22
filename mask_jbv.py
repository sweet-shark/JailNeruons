import torch
import os
import pandas as pd


from MLLM_models import *

model = LLaVA()
model_name = 'llava'
jbv_version = 'jbv'
model.eval()


query_df = pd.read_csv('/path/to/your/JailBreakV_28k/mini_JailBreakV_28K.csv')
attack_success = torch.load(f'/path/to/your/JailBreakV_28k/{model_name}/attack_success.pt')

batch_query_text = query_df["jailbreak_query"]
batch_image_path = [os.path.join('/path/to/your/JailBreakV_28k', path) for path in
                    query_df["image_path"]]
print("Image loaded.")
df_save = query_df
batch_response = [None] * len(batch_image_path)

alpha = 1
layer_idxs = [5,6,7,8,9,]
suffix = '_sorry'
for layer_idx in layer_idxs:
    for idx, index in enumerate(attack_success):
        image_path = batch_image_path[index]
        prompt = batch_query_text[index]
        print(f"{layer_idx}, {idx}/{len(attack_success)}")

        if model_name == 'llava':
            label = model.processor.tokenizer.encode('Sorry')[1] # llava
            target = torch.zeros(1, 128320, device='cuda')
        elif model_name == 'janus':
            label = model.tokenizer.encode('Sorry')[1] # janus
            target = torch.zeros(1, 102400, device='cuda')
        elif model_name == 'qwen':
            label = model.processor.tokenizer.encode('Sorry')[0]
            target = torch.zeros(1, 152064, device='cuda')
            # print(label)
        elif model_name == 'minigpt4':
            label = model.model.llama_tokenizer.encode('Sorry')[0]
            target = torch.zeros(1, 32000, device='cuda')
            # raise
        target[0, label] = 1.0

        mask = model.train_mask(image_path, target, prompt=prompt, layer_idx=layer_idx, alpha=alpha, t = True)
        torch.save(mask, f'./masks/{model_name}/{jbv_version}_{index}_{layer_idx}_alpha{alpha}{suffix}.pt')
    
    