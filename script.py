import numpy as np
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizer, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

d1 = load_dataset("nyu-mll/glue", "sst2")
d1 = d1['train'].select(range(500))
print('type of d1: ', type(d1))
d2 = load_dataset("nyu-mll/glue", "mrpc")
d2 = d2['train'].select(range(500))
d3 = load_dataset("nyu-mll/glue", "wnli")
d3 = d3['train'].select(range(500))

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

# we chose this small model
model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-1M')

def calculate_num_tokens(dataset, tokenizer):
    # calculate number of tokens in the dataset
    num_tokens = 0
    for sample in dataset:
        if isinstance(sample, dict):
            # num_tokens += len(tokenizer.encode(sample['sentence1'])) + len(tokenizer.encode(sample['sentence2']))
            num_tokens += len(tokenizer.encode(sample['text']))

        else:
            # If the sample is not a dict, print its type and content for debugging
            print(f"Unexpected sample type: {type(sample)}")
            print(f"Sample content: {sample}")
            break
    return num_tokens



# Convert datasets to have consistent features
def standardize_dataset(dataset):
    # Keep only the text columns and convert to a common format
    if 'sentence' in dataset.features:
        # SST2 format
        return dataset.remove_columns(['label', 'idx']).rename_column('sentence', 'text')
    else:
        # MRPC and WNLI format
        return dataset.remove_columns(['label', 'idx']).rename_column('sentence1', 'text')

# Standardize each dataset
d1_std = standardize_dataset(d1)
d2_std = standardize_dataset(d2)
d3_std = standardize_dataset(d3)

# Now combine the standardized datasets
combined_dataset = concatenate_datasets([d1_std, d2_std, d3_std])

print("number of tokens in the combined dataset: ", calculate_num_tokens(combined_dataset, tokenizer))



