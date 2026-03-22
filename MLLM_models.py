
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, LlavaNextImageProcessor
from qwen_vl_utils import process_vision_info

import argparse
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import csv
import tqdm
import torch.nn as nn
import sys
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

sys.path.append('../../model_framework/Janus')
sys.path.append('../../model_framework')

from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

from minigpt4.conversation.conversation import CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
import PIL.Image as Image


def normalize(image, mean, std):
    num_channel = len(image)
    mean = torch.tensor(mean, device=image.device)
    std = torch.tensor(std, device=image.device)
    mean = mean.unsqueeze(1).unsqueeze(1) # shape = 3, 1, 1
    std = std.unsqueeze(1).unsqueeze(1)
    for _ in range(num_channel-3):
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    denormalize_image = (image - mean)/std
    return denormalize_image


def denormalize(image, mean, std):
    num_channel = len(image)
    mean = torch.tensor(mean, device=image.device)
    std = torch.tensor(std, device=image.device)
    mean = mean.unsqueeze(1).unsqueeze(1) # shape = 3, 1, 1
    std = std.unsqueeze(1).unsqueeze(1)
    for _ in range(num_channel-3):
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    denormalize_image = image * std + mean
    return denormalize_image

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/minigpt4_llama2_eval.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


class MiniGPT(nn.Module):
    def __init__(self,):
        super(MiniGPT, self).__init__()

        # ========================================
        #             Model Initialization
        # ========================================

        conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
                     'pretrain_vicuna1': CONV_VISION_Vicuna0,
                     'pretrain_llama2': CONV_VISION_LLama2}

        random_number = random.randint(1, 2000)
        random.seed(random_number)
        np.random.seed(random_number)
        torch.manual_seed(random_number)

        cudnn.benchmark = False
        cudnn.deterministic = True

        print('Initializing Chat')
        args = parse_args()
        cfg = Config(args)
        device = 'cuda:{}'.format(args.gpu_id)

        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        # print(model_config)
        # ra
        
        self.model = model_cls.from_config(model_config).to(device)


        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

        self.model = self.model.eval()

        self.CONV_VISION = conv_dict[model_config.model_type]
        self.device = device
        stop_words_ids = [[835], [2277, 29937]]
        stop_words_ids = [torch.tensor(ids).to(self.device) for ids in stop_words_ids]
        # stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        # if stopping_criteria is not None:
        #     self.stopping_criteria = stopping_criteria
        # else:
        #     stop_words_ids = [torch.tensor([2]).to(self.device)]
        #     self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        print('Initialization Finished')
        for param in self.model.parameters():
            param.requires_grad = False
    


    def update_training_msg(self, rnd_idx):
        conv = self.CONV_VISION.copy()
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")

        self.mprompt = []
        self.answers = []
        self.test_target = []
        rr = 0
        with open('./dataset/advbench/harmful_behaviors_2.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if rr == rnd_idx:
                    self.mprompt.append(row['goal'])
                    self.answers.append(row['target'])
                rr += 1
                
        self.q_conv = []
        for test_text in self.mprompt:
            conv_tmp = conv.copy()
            if len(conv_tmp.messages) > 0 and conv_tmp.messages[-1][0] == conv_tmp.roles[0] \
                    and conv_tmp.messages[-1][1][-6:] == '</Img>':  # last message is image.
                conv_tmp.messages[-1][1] = ' '.join([conv_tmp.messages[-1][1], test_text])
            else:
                conv_tmp.append_message(conv_tmp.roles[0], test_text)
            conv_tmp.append_message(conv_tmp.roles[1], None)
            self.q_conv.append(conv_tmp)

        self.conv = []
        self.target_len = []
        self.shift_labels = []
        self.train_num = 1
        for k in range(self.train_num):
            #print(k)
            goal = self.mprompt[k]
            target = self.answers[k]
            conv_ = conv.copy()

            if len(conv_.messages) > 0 and conv_.messages[-1][0] == conv_.roles[0] \
                    and conv_.messages[-1][1][-6:] == '</Img>':  # last message is image.
                conv_.messages[-1][1] = ' '.join([conv_.messages[-1][1], goal])
            else:
                conv_.append_message(conv_.roles[0], goal)
            conv_.append_message(conv_.roles[1], target)
            self.conv.append(conv_)


    def get_context_emb(self, conv, img_list, flag):
        prompt = conv.get_prompt()

        prompt_segs = prompt.split('<ImageHere>')

        seg_tokens = [
            self.model.llama_tokenizer(
                seg , return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        
        inputs_tokens = []
        inputs_tokens.append(seg_tokens[0])
        # inputs_tokens.append( torch.from_numpy(np.ones((1,32))*(-200)).to(self.device) ) #for 224*224 num_Vtokens=32
        inputs_tokens.append(torch.from_numpy(np.ones((1, 64)) * (-200)).to(self.device))  # for 448*448 num_Vtokens=256
        inputs_tokens.append(seg_tokens[1])
        
        # inst_loc = torch.where(seg_tokens[1] == 29962)[1].item()
        # self.prefix_len = []
        # self.prefix_len.append(seg_tokens[0].shape[1] + 64 + inst_loc)
        dtype = inputs_tokens[0].dtype
        inputs_tokens = torch.cat(inputs_tokens, dim=1).to(dtype)

        seg_embs = [self.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        # for seg_emb in seg_embs:
        #     print('seg_emb.shape',seg_emb.shape)
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        
        # for mixed_emb in mixed_embs:
        #     print('mixed_emb.shape',mixed_emb.shape)
        
        mixed_embs = torch.cat(mixed_embs, dim=1)
        # print(mixed_embs.shape)
        # raise
        return mixed_embs, inputs_tokens

    

    def forward(self, inp):
        r"""
        Overridden.

        """

        '''
        prompt, image_position, torch_image = process_image(self.prompt, image=image)

        torch_image = torch_image.to(next(self.model.parameters()).dtype).to(next(self.model.parameters()).device)


        #tokens = self.seq[:self.mask_position + 2].unsqueeze(0)
        #tokens = self.labels[:self.mask_position + 2 + 1].unsqueeze(0)
        #tokens = self.labels[:self.mask_position + 2 + 5].unsqueeze(0)
        tokens = self.labels[:self.mask_position + 2 + 6*3*2].unsqueeze(0)
        #print(tokens)

        #logits = self.model(input_ids=tokens, image=image, pre_image=pre_image)[0]
        logits = self.model(input_ids=tokens, image=torch_image, pre_image=self.pre_image)[0]
        dtype = logits.dtype
        lm_logits = logits.to(torch.float32)
        '''

        images = inp[0]
        k = inp[1]

        image_emb, _ = self.model.encode_img(images)
        image_list = []
        image_list.append(image_emb)

        shift_logits = []

        loss_fct = nn.CrossEntropyLoss(ignore_index=-200)

        loss = 0
        # for k in range(len(self.conv)):
        if 1:
            conv_ = self.conv[k]
            # print('the training conv is', conv_)
            # raise
            target_len_ = self.target_len[k]
            prefix_len_ = self.prefix_len[k]
            shift_labels_ = self.shift_labels[k][:,prefix_len_:]
            # print('label_shape = ', shift_labels_.shape)
            # print('image_embedding_shape = ', image_emb.shape)
            # print(conv_)
            embs, _ = self.get_context_emb(conv_, image_list, True)
            # print('image_conv_shape = ', embs.shape, target_len_)
            max_new_tokens = 300
            max_length = 2000

            current_max_len = embs.shape[1] + max_new_tokens
            if current_max_len - max_length > 0:
                print('Warning: The number of tokens in current conversation exceeds the max length. '
                      'The model will not see the contexts outside the range.')
            begin_idx = max(0, current_max_len - max_length)
            embs = embs[:, begin_idx:]

            outputs = self.model.llama_model(inputs_embeds=embs)
            logits = outputs.logits
            # print(embs.shape,logits.shape, target_len_)
            # raise
            lm_logits = logits[:, prefix_len_:target_len_, :]

            # Shift so that tokens < n predict n
            shift_logits_ = lm_logits[..., :-1, :].contiguous()
            shift_logits.append(shift_logits_)
            # print(shift_logits_.shape, shift_labels_)
            # raise
            loss += loss_fct(shift_logits_.view(-1, shift_logits_.size(-1)), shift_labels_.view(-1))

        return -loss
    
    def inference(self, image_path, prompt, generate=True):
        import torchvision.transforms as transforms
        max_new_tokens = 150
        max_length = 2000
        transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
        conv = self.CONV_VISION.copy()
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        conv_tmp = conv.copy()
        if len(conv_tmp.messages) > 0 and conv_tmp.messages[-1][0] == conv_tmp.roles[0] \
                and conv_tmp.messages[-1][1][-6:] == '</Img>':  # last message is image.
            conv_tmp.messages[-1][1] = ' '.join([conv_tmp.messages[-1][1], prompt])
        else:
            conv_tmp.append_message(conv_tmp.roles[0], prompt)
        conv_tmp.append_message(conv_tmp.roles[1], None)
        self.q_conv= [conv_tmp]
        if isinstance(image_path, Image.Image):
            adv_image = image_path
        else:
            adv_image = Image.open(image_path).convert('RGB')
        adv_img = transform(adv_image)
        adv_img = [adv_img.cuda().unsqueeze(0)]
        image_emb, _ = self.model.encode_img(adv_img[0])  # NOBUG
        image_list = []
        image_list.append(image_emb)

        q_conv = self.q_conv[0]
        embs, _ = self.get_context_emb(q_conv, image_list, False)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                'The self will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]
        if generate:
            num_beams = 1
            top_p = 0.9
            repetition_penalty = 1.05
            length_penalty = 1
            temperature = 1.0
            min_length = 1
            generation_dict = dict(
                inputs_embeds=embs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=True,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=float(temperature),
            )
            output_token = self.model.llama_model.generate(**generation_dict)[0]
            output_text = self.model.llama_tokenizer.decode(output_token, skip_special_tokens=True)

            return output_text
        else:
            outputs = self.model.llama_model(inputs_embeds=embs, output_hidden_states=True)
            return outputs
    
    def validate(self, image, prompt=None, max_new_tokens=100, generate=True):
        image_emb, _ = self.model.encode_img(image[0])
        conv_ = self.conv[0]
        image_list = []
        image_list.append(image_emb)
        embs, _ = self.get_context_emb(conv_, image_list, True)
        max_length = 2000
        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                    'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]
        if generate:
            num_beams = 1
            top_p = 0.9
            repetition_penalty = 1.05
            length_penalty = 1
            temperature = 1.0
            min_length = 1
            generation_dict = dict(
                inputs_embeds=embs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=True,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=float(temperature),
            )
            output_token = self.model.llama_model.generate(**generation_dict)[0]
            
            output_text = self.model.llama_tokenizer.decode(output_token, skip_special_tokens=True)
            return output_text
        else:
            outputs = self.model.llama_model(inputs_embeds=embs, output_hidden_states=True)
            return outputs
    
    def train_mask(self, image_path, target, prompt=None, layer_idx=0, alpha=0.1, lr=8, t=False):
        import torchvision.transforms as transforms
        max_new_tokens = 500
        max_length = 2000
        transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
        conv = self.CONV_VISION.copy()
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        conv_tmp = conv.copy()
        if len(conv_tmp.messages) > 0 and conv_tmp.messages[-1][0] == conv_tmp.roles[0] \
                and conv_tmp.messages[-1][1][-6:] == '</Img>':  # last message is image.
            conv_tmp.messages[-1][1] = ' '.join([conv_tmp.messages[-1][1], prompt])
        else:
            conv_tmp.append_message(conv_tmp.roles[0], prompt)
        conv_tmp.append_message(conv_tmp.roles[1], None)
        self.q_conv= [conv_tmp]
        if isinstance(image_path, Image.Image):
            adv_image = image_path
        else:
            adv_image = Image.open(image_path)
        adv_img = transform(adv_image)
        adv_img = [adv_img.cuda().unsqueeze(0)]
        image_emb, _ = self.model.encode_img(adv_img[0])  # NOBUG
        image_list = []
        image_list.append(image_emb)

        q_conv = self.q_conv[0]
        embs, _ = self.get_context_emb(q_conv, image_list, False)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                'The self will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]
        
        num_steps = 300  #
        mask = nn.Parameter(torch.randn(1,1,4096,device='cuda').to(torch.bfloat16))

        def hook_fn(module, input, output):
            # print( output[0].dtype)
            new_output = output[0] * (1-torch.sigmoid(mask))
            # raise
            
            return (new_output, output[1])
        
        layer_name = f"model.layers.{layer_idx}" 
        layer = dict(self.model.llama_model.named_modules())[layer_name]
        hook = layer.register_forward_hook(hook_fn)
        # optimizer = torch.optim.Adam([mask], lr=lr, eps=1e-4)
        optimizer = torch.optim.SGD([mask], lr=lr,)
        # scaler = GradScaler()
        
        loss_mask_old = 0
        loss_pred_old = 0
        flag = 0
        for step in range(num_steps):
            # if step == 6:
            #     #refresh the optimizer
            #     optimizer = torch.optim.Adam([mask], lr=lr, eps=1e-4)
                
            optimizer.zero_grad()
            # with autocast(device_type='cuda'):
            outputs = self.model.llama_model(inputs_embeds=embs, output_hidden_states=True)
            
            logits = outputs.logits[:,-1]
            scores = F.softmax(logits, dim=1)+1e-7
            loss_mask =  torch.sigmoid(mask).sum()
            # value, idx = torch.sort(logits[0])
            # print(self.processor.decode([2217]))
            # raise
            # print(value[-20:], idx[-20:])
            # print(scores)
            # print(torch.sigmoid(mask))
            # raise
            loss_pred = (target*scores.log()).sum(dim=-1)
            if t:
                loss_pred = -loss_pred
            

            if step > 5:
                loss = loss_pred *2 + alpha/10 * loss_mask
            else:
                loss = loss_pred *2 
                
            loss.backward()
            torch.nn.utils.clip_grad_value_([mask], clip_value=1.0)
            # print(mask.grad.max())
            optimizer.step()

            # scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_value_([mask], clip_value=1.0)
            # scaler.step(optimizer)
            # scaler.update()
            
            print(step, loss.item(), loss_mask.item(), loss_pred.item())
            if loss_mask_old - loss_mask <= 1e-7:
                if loss_pred_old - loss_pred <= 1e-7:
                    if step > 5:
                        flag += 1
                    
            if flag >= 3:
                break 
            loss_mask_old = loss_mask
            loss_pred_old = loss_pred
            # old_loss = loss.item()
            
            # mask = torch.clamp(mask, 0, 1)
        hook.remove()
        print(step, loss_mask_old.item(), loss_mask.item(),\
                        loss_pred.item(), loss_pred_old.item())
        # raise
        return mask
    
    
class LLaVA(nn.Module):
    def __init__(self,):
        super(LLaVA, self).__init__()
        random_number = random.randint(1, 2000)
        random.seed(random_number)
        np.random.seed(random_number)
        torch.manual_seed(random_number)

        cudnn.benchmark = False
        cudnn.deterministic = True
        self.model_path = "/path/to/your/llama3_llava_next_8b"
        self.processor = LlavaNextProcessor.from_pretrained(self.model_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, device_map="auto") 
        for param in self.model.parameters():
            param.requires_grad = False
            
            
    def update_training_msg(self, rnd_idx):
        # raise
        self.mprompt = []
        self.answers = []
        self.test_target = []
        rr = 0
        with open('./dataset/advbench/harmful_behaviors_2.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if rr == rnd_idx:
                    self.mprompt.append(row['goal'])
                    self.answers.append(row['target'])
                rr += 1
                
        self.q_conv = []

        self.shift_labels = []
        self.prefix_len = []
        self.train_num = 1
        for k in range(self.train_num):
            target = self.answers[k]
            test_text = self.mprompt[k]
            conv_tmp = [
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": test_text},
                    {"type": "image"},
                    ],
                },
                {
                    "role": "assistant",
                    "content":[
                        {"type": "text", "text": target+'. First,'}
                    ]
                }
            ]
            image = Image.open('./dataset/benign_images/10000.png')
            prompt = self.processor.apply_chat_template(conv_tmp, add_generation_prompt=True)[:-57]
            # print(prompt[:-57])
            # raise
            prepare_inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)
            # print(prepare_inputs.keys())
            # raise
            self.q_conv.append(prepare_inputs)
            shift_labels_ = self.processor.tokenizer.encode(target+'. First,')
            # print(shift_labels_, )
            # image_pixel = self.processor.image_processor(image)
            denormalize_image = denormalize(prepare_inputs['pixel_values'], self.processor.image_processor.image_mean, self.processor.image_processor.image_std)
            # print(denormalize_image.shape)
            # raise
            self.shift_labels.append(shift_labels_[1:]) #remove the <｜begin▁of▁sentence｜> on index 0
            # print(prepare_inputs['input_ids'], shift_labels_)
            # raise
            return denormalize_image

    def forward(self, inp):
        r"""
        Overridden.
        """

        '''
        prompt, image_position, torch_image = process_image(self.prompt, image=image)

        torch_image = torch_image.to(next(self.model.parameters()).dtype).to(next(self.model.parameters()).device)


        #tokens = self.seq[:self.mask_position + 2].unsqueeze(0)
        #tokens = self.labels[:self.mask_position + 2 + 1].unsqueeze(0)
        #tokens = self.labels[:self.mask_position + 2 + 5].unsqueeze(0)
        tokens = self.labels[:self.mask_position + 2 + 6*3*2].unsqueeze(0)
        #print(tokens)

        #logits = self.model(input_ids=tokens, image=image, pre_image=pre_image)[0]
        logits = self.model(input_ids=tokens, image=torch_image, pre_image=self.pre_image)[0]
        dtype = logits.dtype
        lm_logits = logits.to(torch.float32)
        '''

        images = inp[0]
        k = inp[1]
        prepare_inputs = self.q_conv[k]
        images = normalize(images, self.processor.image_processor.image_mean, self.processor.image_processor.image_std)
        prepare_inputs['pixel_values'] = images
        output = self.model(**prepare_inputs,  )
        shift_labels_ = torch.tensor(self.shift_labels[k], device='cuda')
        num_label = len(shift_labels_)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-200)
        logits = output['logits'][0, -(num_label+1):-1]
        loss = loss_fct(logits, shift_labels_.view(-1))

        return -loss

    def validate(self, images, generate=False):
        conv_tmp = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.mprompt[0]},
                    {"type": "image"},
                    ],
                },]
        image = Image.open('./dataset/benign_images/10000.png')
        prompt = self.processor.apply_chat_template(conv_tmp, add_generation_prompt=True)
        prepare_inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)
        images = normalize(images, self.processor.image_processor.image_mean, self.processor.image_processor.image_std)
        prepare_inputs['pixel_values'] = images
        if generate:
            output = self.model.generate(**prepare_inputs, max_new_tokens=100)
            answer = self.processor.decode(output[0], skip_special_tokens=True)
            print(self.mprompt[0], answer)
            return answer
        else:
            outputs = self.model(**prepare_inputs, output_hidden_states=True)
            return outputs
    
    def inference(self, image_path, prompt=None, generate=True, max_new_tokens=100):
        if prompt is None:
            prompt = self.mprompt[0]
        conv_tmp = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                    ],
                },]
        if isinstance(image_path, Image.Image):
            image = image_path
        else:
            image = Image.open(image_path)
        prompt = self.processor.apply_chat_template(conv_tmp, add_generation_prompt=True)
        prepare_inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)
        if generate:
            output = self.model.generate(**prepare_inputs, max_new_tokens=max_new_tokens)

            generated_ids_trimmed = output[0, len(prepare_inputs.input_ids[0]):]
            # print(output.shape)
            # print(len(prepare_inputs.input_ids[0]))
            # raise
            answer = self.processor.decode(generated_ids_trimmed, skip_special_tokens=True)
            return answer
        else:
            outputs = self.model(**prepare_inputs, output_hidden_states=True)
            return outputs
    
    def train_mask(self, image_path, target, prompt=None, layer_idx=0, alpha=0.1, lr=5, t=False):
        if prompt is None:
            prompt = self.mprompt[0]

        conv_tmp = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                    ],
                },]
        prompt = self.processor.apply_chat_template(conv_tmp, add_generation_prompt=True)
        if isinstance(image_path, str):
            image = Image.open(image_path)
        else:
            image = Image.open('./dataset/benign_images/10000.png')
        prepare_inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)
        # print(len(prepare_inputs['input_ids'][0]))
        # raise
        if isinstance(image_path, torch.Tensor):
            prepare_inputs['pixel_values'] = image_path[0].unsqueeze(0).to(torch.float16)
        
        
        with torch.no_grad():
            prepare_inputs['input_ids'] = torch.cat([prepare_inputs['input_ids'], torch.tensor([[198]], device='cuda')], dim=1)
            prepare_inputs['attention_mask'] = torch.cat([prepare_inputs['attention_mask'], torch.tensor([[1]], device='cuda')], dim=1)
            if target is None:
                outputs = self.model(**prepare_inputs, output_hidden_states=True)
                target_logits = outputs.logits[:,-1]
                target = F.softmax(target_logits, dim=1)
        # print(target.log()*target)
        # raise
        
        num_steps = 200  #
        mask = nn.Parameter(torch.randn(1,1,4096,device='cuda').to(torch.bfloat16))

        def hook_fn(module, input, output):
            # global mask
            new_output = output[0] * (1-torch.sigmoid(mask))
            return (new_output, output[1])
        
        layer_name = f"model.layers.{layer_idx}" 
        layer = dict(self.model.language_model.named_modules())[layer_name]
        hook = layer.register_forward_hook(hook_fn)
        # optimizer = torch.optim.Adam([mask], lr=lr, eps=1e-4)
        optimizer = torch.optim.SGD([mask], lr=lr,)
        # scaler = GradScaler()
        
        loss_mask_old = 0
        loss_pred_old = 0
        flag = 0
        for step in range(num_steps):
            # if step == 6:
            #     #refresh the optimizer
            #     optimizer = torch.optim.Adam([mask], lr=lr, eps=1e-4)
                
            optimizer.zero_grad()
            # with autocast(device_type='cuda'):
            outputs = self.model(**prepare_inputs, output_hidden_states=True)
            
            logits = outputs.logits[:,-1]
            scores = F.softmax(logits, dim=1)+1e-7
            loss_mask =  torch.sigmoid(mask).sum()
            # value, idx = torch.sort(logits[0])
            # print(self.processor.decode([2217]))
            # raise
            # print(value[-20:], idx[-20:])
            # print(scores)
            # print(torch.sigmoid(mask))
            # raise
            loss_pred = (target*scores.log()).sum(dim=-1)
            if t:
                loss_pred = -loss_pred
            

            if step > 5:
                loss = loss_pred * 2 + alpha/10 * loss_mask
            else:
                loss = loss_pred * 2
                
            loss.backward()
            torch.nn.utils.clip_grad_value_([mask], clip_value=5.0)
            # print(mask.grad.max())
            optimizer.step()

            # scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_value_([mask], clip_value=1.0)
            # scaler.step(optimizer)
            # scaler.update()
            
            print(step, loss.item(), loss_mask.item(), loss_pred.item())
            if torch.abs(loss_mask_old-loss_mask) <= 1e-7:
                if torch.abs(loss_pred - loss_pred_old) <= 1e-7:
                    flag += 1
                    
            if flag >= 3:
                break 
            loss_mask_old = loss_mask
            loss_pred_old = loss_pred
            # old_loss = loss.item()
            
            # mask = torch.clamp(mask, 0, 1)
        hook.remove()
        print(step, loss_mask_old.item(), loss_mask.item(),\
                        loss_pred.item(), loss_pred_old.item())
        # raise
        return mask

class Janus(nn.Module):
    def __init__(self,):
        super(Janus, self).__init__()
        random_number = random.randint(1, 2000)
        random.seed(random_number)
        np.random.seed(random_number)
        torch.manual_seed(random_number)

        cudnn.benchmark = False
        cudnn.deterministic = True
        self.model_path = "/path/to/your/Janus-pro-7b"
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()
        for param in self.vl_gpt.parameters():
            param.requires_grad = False
            
    def update_training_msg(self, rnd_idx):
        # raise
        self.mprompt = []
        self.answers = []
        self.test_target = []
        rr = 0
        with open('./dataset/advbench/harmful_behaviors_2.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if rr == rnd_idx:
                    self.mprompt.append(row['goal'])
                    self.answers.append(row['target'])
                rr += 1
                
        self.q_conv = []
        self.conv = []
        self.target_len = []
        self.shift_labels = []
        self.prefix_len = []
        self.train_num = 1
        for k in range(self.train_num):
            target = self.answers[k]
            test_text = self.mprompt[k]
            conv_tmp = [{
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{test_text}",
                "images": ['./dataset/benign_images/10000.png'],
            },
            {"role": "<|Assistant|>", "content":f"{target}"+'First,'},]
            pil_images = load_pil_images(conv_tmp)
            prepare_inputs = self.vl_chat_processor(
                conversations=conv_tmp, images=pil_images, force_batchify=True
            ).to(self.vl_gpt.device)
            # print(prepare_inputs['input_ids'])
            self.q_conv.append(prepare_inputs)
            shift_labels_ = self.tokenizer.encode(target+'First,')
            # print(shift_labels_)
            # raise
            self.shift_labels.append(shift_labels_[1:]) #remove the <｜begin▁of▁sentence｜> on index 0

            

    def forward(self, inp):
        r"""
        Overridden.

        """

        '''
        prompt, image_position, torch_image = process_image(self.prompt, image=image)

        torch_image = torch_image.to(next(self.model.parameters()).dtype).to(next(self.model.parameters()).device)


        #tokens = self.seq[:self.mask_position + 2].unsqueeze(0)
        #tokens = self.labels[:self.mask_position + 2 + 1].unsqueeze(0)
        #tokens = self.labels[:self.mask_position + 2 + 5].unsqueeze(0)
        tokens = self.labels[:self.mask_position + 2 + 6*3*2].unsqueeze(0)
        #print(tokens)

        #logits = self.model(input_ids=tokens, image=image, pre_image=pre_image)[0]
        logits = self.model(input_ids=tokens, image=torch_image, pre_image=self.pre_image)[0]
        dtype = logits.dtype
        lm_logits = logits.to(torch.float32)
        '''

        images = inp[0]
        k = inp[1]
        prepare_inputs = self.q_conv[k]
        prepare_inputs['pixel_values'] = images.unsqueeze(0).to(torch.bfloat16)
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        outputs = self.vl_gpt.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_hidden_states=True,  
            output_logits = True,
            
        )
        
        shift_labels_ = torch.tensor(self.shift_labels[k], device='cuda')
        num_label = len(shift_labels_)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-200)

        logits = outputs.logits[:, -(num_label+1):-1]
        # uu = logits.view(-1, logits.size(-1))
        # print(uu.shape)
        # raise
        
        loss = loss_fct(logits.view(-1, logits.size(-1)), shift_labels_.view(-1))
        # print(loss)
        # raise
        return -loss
    
    def validate(self, images, generate=True, prompt=None):
        if prompt is None:
            prompt = self.mprompt[0]
        conv_tmp = [{
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{prompt}",
                "images": ['./dataset/benign_images/1233.jpg'],
            },
            {"role": "<|Assistant|>", "content": ""},]
        pil_images = load_pil_images(conv_tmp)
        prepare_inputs = self.vl_chat_processor(
            conversations=conv_tmp, images=pil_images, force_batchify=True
        ).to(self.vl_gpt.device)
        
        prepare_inputs['pixel_values'] = images[0].unsqueeze(0).to(torch.bfloat16)
        self.imagepos = torch.where(prepare_inputs['input_ids'] == 100594)[1][-1]
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        if generate:
            outputs = self.vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
            )
        
            answer = self.tokenizer.decode(outputs.sequences[0].cpu().tolist(), skip_special_tokens=True)
            return answer
        else:
            outputs = self.vl_gpt.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                output_hidden_states=True,  
                output_logits = True,
            )
            return outputs
    
    def train_multilayer_mask(self, image_path, target, prompt=None, layer_idx=[5, 8, 9, 11, 16], alpha=1):
        if prompt is None:
            prompt = self.mprompt[0]
        
        target = target.clone().detach()
        target = F.softmax(target, dim=1)
            
        if isinstance(image_path, str) or isinstance(image_path, Image.Image):
            conv_tmp = [{
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{prompt}",
                    "images": [image_path],
                },
                {"role": "<|Assistant|>", "content": ""},]
        else:
            conv_tmp = [{
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{prompt}",
                    "images": ['./dataset/benign_images/1233.jpg'],
                },
                {"role": "<|Assistant|>", "content": ""},]
        pil_images = load_pil_images(conv_tmp)
        prepare_inputs = self.vl_chat_processor(
            conversations=conv_tmp, images=pil_images, force_batchify=True
        ).to(self.vl_gpt.device)
        if isinstance(image_path, torch.Tensor):
            prepare_inputs['pixel_values'] = image_path[0].unsqueeze(0).to(torch.bfloat16)
        # self.imagepos = torch.where(prepare_inputs['input_ids'] == 100594)[1][-1]
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        num_steps = 200  
        
        optimizers = []
        masks = []
        hooks = []
        for j in range(len(layer_idx)):
            maski = nn.Parameter(torch.ones(1,1,4096,device='cuda').to(torch.bfloat16))
            masks.append(maski)
            def hook_fn0(module, input, output):
                new_output = output[0] * torch.sigmoid(maski)
                return (new_output, output[1])
            layer_name = f"model.layers.{layer_idx[0]}" 
            layer = dict(self.vl_gpt.language_model.named_modules())[layer_name]
            hook = layer.register_forward_hook(hook_fn0)
            hooks.append(hook)
            optimizer = torch.optim.Adam([maski], lr=0.1)
            optimizers.append(optimizer)
        
        for step in range(num_steps):

            optimizer.zero_grad()
            outputs = self.vl_gpt.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                output_logits = True,
            )
            logits = outputs.logits[:,-1]
            scores = F.softmax(logits, dim=1)
            l1_loss = sum(torch.sigmoid(mask).sum() for mask in masks)
            loss = -(target*scores.log()).sum(dim=-1).mean() + alpha * l1_loss
            loss.backward()
            # print(mask.grad)
            # raise
            optimizer.step()
            # print()
            print(step, -(target*scores.log()).sum(dim=-1).mean(), l1_loss/len(layer_idx))
            # mask = torch.clamp(mask, 0, 1)
        hook.remove()
        
        return masks
    
    def train_mask(self, image_path, target, prompt=None, layer_idx=0, alpha=1, lr=0.5, t=False):
        if prompt is None:
            prompt = self.mprompt[0]
        
        # print(target.shape)
        # raise
        # target = target.clone().detach()
        # target = F.softmax(target, dim=1)
            
        if isinstance(image_path, str) or isinstance(image_path, Image.Image):
            conv_tmp = [{
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{prompt}",
                    "images": [image_path],
                },
                {"role": "<|Assistant|>", "content": ""},]
        else:
            conv_tmp = [{
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{prompt}",
                    "images": ['./dataset/benign_images/1233.jpg'],
                },
                {"role": "<|Assistant|>", "content": ""},]
        pil_images = load_pil_images(conv_tmp)
        prepare_inputs = self.vl_chat_processor(
            conversations=conv_tmp, images=pil_images, force_batchify=True
        ).to(self.vl_gpt.device)
        if isinstance(image_path, torch.Tensor):
            prepare_inputs['pixel_values'] = image_path[0].unsqueeze(0).to(torch.bfloat16)
        # self.imagepos = torch.where(prepare_inputs['input_ids'] == 100594)[1][-1]
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        num_steps = 200  #
        # global mask
        mask = nn.Parameter(torch.randn(1,1,4096,device='cuda').to(torch.bfloat16))

        def hook_fn(module, input, output):
            # global mask
            new_output = output[0] * (1-torch.sigmoid(mask))
            return (new_output, output[1])
        
        layer_name = f"model.layers.{layer_idx}" 
        layer = dict(self.vl_gpt.language_model.named_modules())[layer_name]
        hook = layer.register_forward_hook(hook_fn)
        optimizer = torch.optim.Adam([mask], lr=lr, )
        
        loss_mask_old = 0
        loss_pred_old = 0
        flag = 0
        for step in range(num_steps):

            optimizer.zero_grad()
            outputs = self.vl_gpt.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                output_logits = True,
            )
            logits = outputs.logits[:,-1]
            scores = F.softmax(logits, dim=1)
            loss_mask = alpha/10 * torch.sigmoid(mask).sum()
            loss_pred = (target*scores.log()).sum(dim=-1).mean()
            if t:
                loss_pred = -loss_pred
            loss = loss_pred + loss_mask
            loss.backward()
            # print(mask.grad)
            # raise
            optimizer.step()
            # print()
            print(step, loss.item(), loss_mask.item(), loss_pred.item())
            if loss_mask_old - loss_mask <= 1e-7:
                if loss_pred_old - loss_pred <= 1e-7:
                    flag += 1
            if flag >= 5:
                break# early stop

            loss_mask_old = loss_mask
            loss_pred_old = loss_pred
            # old_loss = loss.item()
            
            # mask = torch.clamp(mask, 0, 1)
        hook.remove()
        
        return mask
    
    def inference(self, image_path, prompt, generate=True, max_new_tokens=100):
        conv_tmp = [{
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{prompt}",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""},]
        pil_images = load_pil_images(conv_tmp)
        prepare_inputs = self.vl_chat_processor(
            conversations=conv_tmp, images=pil_images, force_batchify=True
        ).to(self.vl_gpt.device)
        self.imagepos = torch.where(prepare_inputs['input_ids'] == 100594)[1][-1]
        # print(self.imagepos)
        # raise
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        if generate:
            outputs = self.vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
            )
            # print(outputs.sequences.shape)
            # raise
            answer = self.tokenizer.decode(outputs.sequences[0].cpu().tolist(), skip_special_tokens=True)
            return answer
        else:
            outputs = self.vl_gpt.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                output_hidden_states=True,  
                output_logits = True,
            )
            return outputs
    
    
class Qwen_vl(nn.Module):
    def __init__(self,):
        
        from modelscope import Qwen2VLForConditionalGeneration, AutoProcessor
        
        super(Qwen_vl, self).__init__()
        random_number = random.randint(1, 2000)
        random.seed(random_number)
        np.random.seed(random_number)
        torch.manual_seed(random_number)

        cudnn.benchmark = False
        cudnn.deterministic = True
        self.model_path = "/path/to/your/Qwen_vl_instruct"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        for param in self.model.parameters():
            param.requires_grad = False
    
    
    def my_process(self, ori_image):
        resized_height, resized_width = ori_image.shape[2:]
        channel = ori_image.shape[1]
        grid_t = ori_image.shape[0] // self.processor.image_processor.temporal_patch_size
        grid_h, grid_w = resized_height // self.processor.image_processor.patch_size, resized_width // self.processor.image_processor.patch_size
        ori_image = ori_image.reshape(
            grid_t,
            self.processor.image_processor.temporal_patch_size,
            channel,
            grid_h // self.processor.image_processor.merge_size,
            self.processor.image_processor.merge_size,
            self.processor.image_processor.patch_size,
            grid_w // self.processor.image_processor.merge_size,
            self.processor.image_processor.merge_size,
            self.processor.image_processor.patch_size,
        )
        # print(resized_height, image_processor.patch_size, resized_width)
        ori_image = ori_image.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = ori_image.reshape(
            grid_t * grid_h * grid_w, channel * self.processor.image_processor.temporal_patch_size * self.processor.image_processor.patch_size * self.processor.image_processor.patch_size
        )
        return flatten_patches
    
    def update_training_msg(self, rnd_idx):
        # raise
        self.mprompt = []
        self.answers = []
        self.test_target = []
        rr = 0
        with open('./dataset/advbench/harmful_behaviors_2.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if rr == rnd_idx:
                    self.mprompt.append(row['goal'])
                    self.answers.append(row['target'])
                rr += 1
                
        self.q_conv = []

        self.shift_labels = []
        self.prefix_len = []
        self.train_num = 1
        for k in range(self.train_num):
            target = self.answers[k]
            test_text = self.mprompt[k]
            
            conv_tmp = [
                {
                "role": "user",
                "content": [
                    {
                            "type": "image",
                            "image": "./dataset/benign_images/10000.png",
                        },
                    {       "type": "text", 
                            "text": test_text
                     },

                    ],
                },
                {
                    "role": "assistant",
                    "content":[
                        {   "type": "text", 
                            "text": target+'. First, '
                        }
                    ]
                }
            ]
            text = self.processor.apply_chat_template(
                conv_tmp, tokenize=False, add_generation_prompt=True
            )[:-33]
            image_inputs, video_inputs = process_vision_info(conv_tmp)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                # return_dict_in_generate=True,
            )
            inputs = inputs.to("cuda")
            denormalize_image = denormalize(inputs['pixel_values'], self.processor.image_processor.image_mean, self.processor.image_processor.image_std)

            self.q_conv.append(inputs)
            shift_labels_ = self.processor.tokenizer.encode(target+'. First, ')

            self.shift_labels.append(shift_labels_) #remove the <｜begin▁of▁sentence｜> on index 0

            return denormalize_image

    def forward(self, inp):
        r"""
        Overridden.
        """

        '''
        prompt, image_position, torch_image = process_image(self.prompt, image=image)

        torch_image = torch_image.to(next(self.model.parameters()).dtype).to(next(self.model.parameters()).device)


        #tokens = self.seq[:self.mask_position + 2].unsqueeze(0)
        #tokens = self.labels[:self.mask_position + 2 + 1].unsqueeze(0)
        #tokens = self.labels[:self.mask_position + 2 + 5].unsqueeze(0)
        tokens = self.labels[:self.mask_position + 2 + 6*3*2].unsqueeze(0)
        #print(tokens)

        #logits = self.model(input_ids=tokens, image=image, pre_image=pre_image)[0]
        logits = self.model(input_ids=tokens, image=torch_image, pre_image=self.pre_image)[0]
        dtype = logits.dtype
        lm_logits = logits.to(torch.float32)
        '''

        images = inp[0]
        k = inp[1]
        prepare_inputs = self.q_conv[k]
        images = normalize(images, self.processor.image_processor.image_mean, self.processor.image_processor.image_std)
        flatten_images = self.my_process(images)
        # print(images.shape)
        # raise
        prepare_inputs['pixel_values'] = flatten_images
        output = self.model(**prepare_inputs,  )

        shift_labels_ = torch.tensor(self.shift_labels[k], device='cuda')
        num_label = len(shift_labels_)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-200)


        logits = output['logits'][0, -(num_label+1):-1]
        
        loss = loss_fct(logits, shift_labels_.view(-1))
        # print(loss)
        # raise
        return -loss

    def validate(self, images, generate=False):
        conv_tmp = [
                {
                "role": "user",
                "content": [
                    {
                            "type": "image",
                            "image": "./dataset/benign_images/10000.png",
                        },
                    {       "type": "text", 
                            "text": self.mprompt[0]
                     },

                    ],
                },

            ]
        text = self.processor.apply_chat_template(conv_tmp, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(conv_tmp)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            # return_dict_in_generate=True,
        )
        inputs = inputs.to("cuda")
        images = normalize(images, self.processor.image_processor.image_mean, self.processor.image_processor.image_std)
        flatten_images = self.my_process(images)
        inputs['pixel_values'] = flatten_images
        if generate:
            generated_ids = self.model.generate(**inputs, max_new_tokens=120,)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            # print(self.mprompt[0], output_text)
            # raise
            return output_text
        else:
            output = self.model(**inputs,  output_hidden_states=True)
            return output
        
        
    def inference(self, image_path, prompt, generate=True):
        conv_tmp = [
                {
                "role": "user",
                "content": [
                    {
                            "type": "image",
                            "image": image_path,
                        },
                    {       "type": "text", 
                            "text": prompt,
                     },

                    ],
                },

            ]
        text = self.processor.apply_chat_template(conv_tmp, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(conv_tmp)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            # return_dict_in_generate=True,
        )
        # print(inputs['attention_mask'].shape, inputs['input_ids'].shape, inputs['pixel_values'].shape, )
        # raise
        inputs = inputs.to("cuda")
        if generate:
            generated_ids = self.model.generate(**inputs, max_new_tokens=120,)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]
        else:
            output = self.model(**inputs,  output_hidden_states=True)
            return output
            
    def train_mask(self, image_path, target, prompt=None, layer_idx=0, alpha=1, lr=0.2, t=False):
        if prompt is None:
            prompt = self.mprompt[0]
        
        # print(target.shape)
        # raise
        # target = target.clone().detach()
        # target = F.softmax(target, dim=1)
            
        if isinstance(image_path, str) or isinstance(image_path, Image.Image):
            conv_tmp = [
                {
                "role": "user",
                "content": [
                    {
                            "type": "image",
                            "image": image_path,
                        },
                    {       "type": "text", 
                            "text": prompt,
                     },

                    ],
                },

            ]
        else:
            conv_tmp = [{
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{prompt}",
                    "images": ['./dataset/benign_images/1233.jpg'],
                },
                {"role": "<|Assistant|>", "content": ""},]
            
            
        text = self.processor.apply_chat_template(conv_tmp, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(conv_tmp)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            # return_dict_in_generate=True,
        )
        inputs = inputs.to("cuda")
        
        num_steps = 200  #
        mask = nn.Parameter(torch.randn(1,1,3584,device='cuda').to(torch.bfloat16))

        def hook_fn(module, input, output):
            # print(output[0].shape)
            # raise
            new_output = output[0] * (1-torch.sigmoid(mask))
            return (new_output, output[1])
        # print(self.model)
        # raise
        layer_name = f"layers.{layer_idx}" 
        # print(dict(self.model.model.named_modules()).keys())
        layer = dict(self.model.model.named_modules())[layer_name]
        hook = layer.register_forward_hook(hook_fn)
        optimizer = torch.optim.Adam([mask], lr=lr, )
        
        loss_mask_old = 0
        loss_pred_old = 0
        flag = 0
        for step in range(num_steps):

            optimizer.zero_grad()
            outputs = self.model(**inputs,  output_hidden_states=True)
            logits = outputs.logits[:,-1]
            scores = F.softmax(logits, dim=1)
            loss_mask = alpha/15 * torch.sigmoid(mask).sum()
            loss_pred = (target*scores.log()).sum(dim=-1).mean()
            if t:
                loss_pred = -loss_pred
                
            if step > 5:
                loss = loss_pred + loss_mask
            else:
                loss = loss_pred
            loss.backward()
            # print(mask.grad)
            # raise
            optimizer.step()
            # print()
            print(step, loss.item(), loss_mask.item(), loss_pred.item())
            if loss_mask_old - loss_mask <= 1e-7:
                if loss_pred_old - loss_pred <= 1e-7:
                    flag += 1
            if flag >= 7:
                break# early stop

            loss_mask_old = loss_mask
            loss_pred_old = loss_pred
            # old_loss = loss.item()
            
            # mask = torch.clamp(mask, 0, 1)
        hook.remove()
        
        return mask