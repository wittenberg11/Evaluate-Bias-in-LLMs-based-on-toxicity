# Evaluate Bias in Chinese LLMs based on toxicity

## Note
This work is based on paper: TrustGPT: A Benchmark for Trustworthy and Responsible Large Language Models

Our aim is to evaluate bias in Chinese large language models, by compare the toxicity value generated between different groups.

## Usage Instruction
**DATA:** Our data is in file data_trust_gpt_prompt.json ,which serves as input into LLMs.

**Model Reasoning:** Run file "generate_toxtic_content.py" to generate the toxic content. For example, run:
```
python generate_toxtic_content.py --model_name 'chatglm3-6b'
```
The generated result are in folder /experiment in json format.


**Calculate Statistics:**  You should have a Perspective API key and add it in folder /config/configuration.json. "cal_statistics.py" are used to calculate statistics. It first fitler the sample that LLM refused to answer, by search if there are keywords like "语言模型","AI" in the answer. Then, we use Perspective API to calculate the toxicity value of each answer and save it in json format. Finally, three metrics: average toxicity value, standard deviation, mann_whitney value are used to measure if there are bias across different groups.
For example, run:
```
python cal_statistics.py --file_path '/experiment/chatglm3-6b_bad.json'  ## which is your result path during model reasoning.
```

## Model
We test five models: Baichuan2-7B-Chat,Qwen-7B-Chat,Yi-6B-Chat,chatglm3-6b,internlm-chat-7b 

Table: Parameter sizes of eight models

| Model              | Para. |
|--------------------|-------|
| Baichuan2       | 7b     |
| Qwen         | 7b   |
| Yi        | 6b   |
| chatglm3      | 6b   |
| internlm       | 7b    |


## Dataset

Our dataset is based on SOCIAL CHEMISTRY 101. This dataset encompasses specific descriptions of social norms, relevant attributes (e.g., legal or illegal), and human judgments (e.g., good or bad) of social norms. Social norms are shared standards of acceptable behavior by groups. A social norm consisting of its actions and human judgement. We selected this dataset because it incorporates social norms with crowd-sourced human evaluation labels, allowing for convenient assessment of the ethical quality associated with a specific action.

We extracted 500 entries labeled as "bad" in the rot-judgment, and translated these into Chinese using the Baidu Translate API.



### API Configuration
Before using Perspective API to obtain toxicity value, you should apply for its api keys: [link](https://developers.perspectiveapi.com/s/?language=en_US)  
If you want to evaluate OpenAI model, you should apply for its api keys: [link](https://platform.openai.com/account/api-keys)  
Then you should put them in `config/configuration.json`:
```
{
"gpt-3.5 API key": your openai api key,
"Perspective API key": your perspective api key
}
```

### Metrics
We implemented four metrics: Std (`utils/metric/Std.py`), RtA (`utils/metric/RtA.py`), toxicity value (`utils/metric/Toxicity.py`) and Mann-Whitney (`utils/metric/Mann-Whitney.py`).  
RtA:
```
list type: RtA_metric(data_list=['As an AI language model, ...', 'I am sorry, ...'])
file type: RtA_metric(filename, filetype)
```
Toxicity value: 
```
get_toxicity_value('Boys are not good at math.')
output: 0.3811502
```
Std:
```
get_Std({'race': ['white', 'black']}, {'black': [0.1, 0.2, 0.3], 'white': [0.3, 0.4, 0.5]})
output: {'race': {'avg': Decimal('0.3'), 'std': Decimal('0.1')}}
```

Mann-Whitney (you should replace the ramdom floats lists with your toxicity list of each group):
```
random_floats_1 = [random.random() for _ in range(20)]
random_floats_2 = [random.random() for _ in range(20)]
random_floats_3 = [random.random() for _ in range(20)]
mann_whitney({'black':random_floats_1, 'white': random_floats_2, 'asian': random_floats_3})
```


