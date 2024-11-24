#!/bin/bash

source stats/bin/activate

subsets=("math" "physics" "cs" "biology" "chemistry")
prompt_types=("iccl" "random" "all_hard")

for subset in "${subsets[@]}"; do
    for prompt_type in "${prompt_types[@]}"; do
        echo "Evaluating for subset: $subset / prompt type: $prompt_type"
        python query_sglang.py \
            --model 'llama2' \
            --subset "$subset" \
            --prompt_type "$prompt_type"
        
        sleep 2
    done
done
