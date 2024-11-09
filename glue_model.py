import pandas as pd
from datasets import load_dataset
from transformers import BertTokenizer, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

sst2_dataset = load_dataset("nyu-mll/glue", "sst2")
sst2_dataset = sst2_dataset["train"].select(range(1000))

mrpc_dataset = load_dataset("nyu-mll/glue", "mrpc")
mrpc_dataset = mrpc_dataset["train"].select(range(1000))

qnli_dataset = load_dataset("nyu-mll/glue", "qnli")
qnli_dataset = qnli_dataset["train"].select(range(1000))

model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-1M')
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

def calculate_num_tokens(dataset, tokenizer):
    # calculate number of tokens in the dataset
    num_tokens = 0
    for sample in dataset:
        if isinstance(sample, dict):
            num_tokens += len(tokenizer.encode(sample['premise'])) + len(tokenizer.encode(sample['hypothesis']))
        else:
            # If the sample is not a dict, print its type and content for debugging
            print(f"Unexpected sample type: {type(sample)}")
            print(f"Sample content: {sample}")
            break

# combine all datasets
combined_dataset = pd.concat([sst2_dataset, mrpc_dataset, qnli_dataset])



