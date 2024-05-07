from utils.dataset.load_dataset import LoadDataset
from utils.prompt.prompt_object import PromptObject

from bias.chatgpt import get_res_bias,call_chatgpt
from utils.metric.Mann_Whitney import *
from utils.metric.RtA import *
from utils.metric.Std import *
from utils.metric.Toxicity import get_toxicity_value,retry_get_toxicity_value
from utils.translate.translate import BaiduAPI
import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel
from transformers.generation.utils import GenerationConfig

def load_data_and_translate(size,path):
    '''
    1：加载数据集
    2：调用翻译 API 翻译成中文
    3: 调用 prompt 模板
    3：保存到 json 文件中
    '''
    if os.path.exists(path):  ## 如果存在，则直接读取 
        with open(path, 'r', encoding='utf-8') as file:
            bias_data = json.load(file)
        print("------- 文件已存在，直接读取-------")
        return bias_data
    
    print("--------文件不存在，新建文件----------")
    load_dataset = LoadDataset()
    bias_data, all_data = load_dataset.preprocess_social101(return_type='bias', shuffle=True, size=size)
    print(all_data[0])

    baidu_api = BaiduAPI()
    ## 翻译成中文 
    result = []
    for item in bias_data:
        item_chinese = baidu_api.translate(item)
        result.append(item_chinese)
    ## 调用 prompt 模板 
    prompt_cons = PromptObject(prompt_type='default', prompt_section='bias', data_item=result)
    bias_data = prompt_cons.construct_default_prompt(prompt_words='有害的' , groups=["男性","女性","中国人","外国人","广东人","新疆人","东北人"])
    ## 保存文件 
    json_file_path = path
    with open(json_file_path, 'w', encoding='utf-8') as file:
        json.dump(bias_data,file,indent=4,ensure_ascii=False)
    return bias_data

## Yi-6B-chat
## Qwen-7B-chat
## chatglm3-6b
## baichuan2-7b-chat

class LLM:
    def __init__(self,model_name):
        self.model_name = model_name

        ## 加载模型 
        if model_name == "Baichuan2-7B-Chat":
            tokenizer = AutoTokenizer.from_pretrained("/mntcephfs/data/med/chenghao/models/Baichuan2-7B-Chat/", use_fast=False, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained("/mntcephfs/data/med/chenghao/models/Baichuan2-7B-Chat/", device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
            model.generation_config = GenerationConfig.from_pretrained("/mntcephfs/data/med/chenghao/models/Baichuan2-7B-Chat/")
        elif model_name == "Qwen-7B-Chat":
            tokenizer = AutoTokenizer.from_pretrained("/mntcephfs/data/med/fanyaxin/Qwen-7B-Chat", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained("/mntcephfs/data/med/fanyaxin/Qwen-7B-Chat", device_map="auto", trust_remote_code=True).eval()
        elif model_name == "Yi-6B-Chat":
            tokenizer = AutoTokenizer.from_pretrained("/mntcephfs/data/med/chenghao/models/Yi-6B-Chat", use_fast=False)
            model = AutoModelForCausalLM.from_pretrained("/mntcephfs/data/med/chenghao/models/Yi-6B-Chat",device_map="auto",torch_dtype=torch.float16).eval()
        elif model_name == "chatglm3-6b":
            tokenizer = AutoTokenizer.from_pretrained("/mntcephfs/data/med/zhanghongbo/MOSS/junying_models/chatglm3-6b", trust_remote_code=True)
            model = AutoModel.from_pretrained("/mntcephfs/data/med/zhanghongbo/MOSS/junying_models/chatglm3-6b", trust_remote_code=True).half().cuda().eval()
        elif model_name == "internlm-chat-7b":
            tokenizer = AutoTokenizer.from_pretrained("/mntcephfs/data/med/chenghao/models/internlm-chat-7b", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained("/mntcephfs/data/med/chenghao/models/internlm-chat-7b", torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()

        self.model = model
        self.tokenizer = tokenizer
    
    def gen_llm(self,query):
        if self.model_name == "Baichuan2-7B-Chat":
            messages = []
            messages.append({"role": "user", "content": query})
            return self.model.chat(self.tokenizer, messages)
        elif self.model_name == "Yi-6B-Chat":
            messages = []
            messages.append({"role": "user", "content": query})
            input_ids = self.tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
            output_ids = self.model.generate(input_ids.to('cuda'))
            response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            return response
        elif self.model_name == "chatglm3-6b" or self.model_name == "internlm-chat-7b" or self.model_name == "Qwen-7B-Chat":
            response, history = self.model.chat(self.tokenizer, query, history=[])
            return response

def main(model_name):
    ## 加载数据
    bias_data = load_data_and_translate(size=200,path="/mntcephfs/lab_data/ganruoli/TrustGPT/data_trust_gpt_prompt.json")
    print("----------- 数据加载完毕 ------------")
    ## 加载模型
    llm = LLM(model_name=model_name)
    print("----------- 模型加载完毕 ------------")

    ## 大模型推理
    total_res = {}
    for group in bias_data.keys():
        group_res = []
        for item in bias_data[group]:
            result={}
            result["query"] = item
            answer = llm.gen_llm(item)
            print(answer)
            result["answer"] = answer
            # result["toxicity"] = retry_get_toxicity_value(answer)
            group_res.append(result)
        total_res[group] = group_res
    
    experiment_result_path = "/mntcephfs/lab_data/ganruoli/TrustGPT/experiment/"+ model_name +"_bad.json"
    with open(experiment_result_path, 'w', encoding='utf-8') as file:
        json.dump(total_res,file,indent=4,ensure_ascii=False)


## Baichuan2-7B-Chat
## Qwen-7B-Chat
## Yi-6B-Chat
## chatglm3-6b
## internlm-chat-7b

# main("Qwen-7B-Chat")
# main("Baichuan2-7B-Chat")
# main("Yi-6B-Chat")
# main("chatglm3-6b")
# main("internlm-chat-7b")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of sft')

    parser.add_argument('--model_name',  type=str)
    args = parser.parse_args()
    model_name = args.model_name
    main(model_name)

        
