import argparse
import sglang as sgl
from sglang import OpenAI, Anthropic, Runtime, assistant, gen, set_default_backend, system, user
import json
import pandas as pd

from sys_prompts import SYS_PROMPT


# Function adapted from: https://github.com/ChuyueSun/Clover/blob/main/clover/clover.py
@sgl.function
def query(s, model, question, answer, subset):
    s += system(SYS_PROMPT)
    s += user(question)
    model_answer = ""

    with s.copy() as tmp:
        tmp += assistant(gen("answer", max_tokens=4096, temperature=0.3))
        model_answer = tmp["answer"]
    
    print(f"Model answer: {model_answer}")
    print(f"Answer: {answer}")
    df = pd.read_csv("iccl_results.csv")
    cur_correct_count = df[(df["model"] == model) & (df["subset"] == subset)]["correct_count"].values[0]
    cur_total_count = df[(df["model"] == model) & (df["subset"] == subset)]["total_count"].values[0]
    if model_answer == answer:
        update_stats(model, subset, int(cur_correct_count+1), int(cur_total_count+1))
    else:
        update_stats(model, subset, cur_correct_count, int(cur_total_count+1))


def update_stats(model, subset, cur_correct_count, total_count):
    results_file = "iccl_results.csv"
    df = pd.read_csv(results_file)
    mask = (df['model'] == model) & (df['subset'] == subset)
    
    if mask.any():
        df.loc[mask, 'correct_count'] = cur_correct_count
        df.loc[mask, 'total_count'] = total_count
    else:
        new_test_result = pd.DataFrame([{
            "model": model,
            "subset": subset,
            "correct_count": cur_correct_count,
            "total_count": total_count
        }])
        df = pd.concat([df, new_test_result], ignore_index=True)
    
    df.to_csv(results_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on MMLU dataset")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--subset", type=str, required=True)
    args = parser.parse_args()
    
    # Model name examples: gpt-3.5-turbo, claude-3-haiku-20240307, meta-llama/Llama-3.2-1B-Instruct, meta-llama/Llama-2-7b-chat-hf
    if args.model.startswith("gpt"):
        set_default_backend(OpenAI(args.model))
    elif args.model.startswith("claude"):
        set_default_backend(Anthropic(args.model))
    elif args.model.startswith("llama3.2"):
        runtime = Runtime(model_path="meta-llama/Llama-3.2-1B-Instruct")
        set_default_backend(runtime)
    elif args.model.startswith("llama2"):
        runtime = Runtime(model_path="meta-llama/Llama-2-7b-chat-hf")
        set_default_backend(runtime)
    else:
        raise ValueError("Invalid model name")

    # Get ICCL prompts
    with open("iccl_prompts.json", "r") as f:
        iccl_prompts = json.load(f)
    
    for idx in iccl_prompts[args.subset]:
        question, answer = iccl_prompts[args.subset][idx]['question'], str(iccl_prompts[args.subset][idx]['answer'])
        query(args.model, question, answer, args.subset)
