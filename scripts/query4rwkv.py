import os
os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # if '1' then use CUDA kernel for seq mode (much faster)
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

import sys
import argparse
import torch
from huggingface_hub import hf_hub_download
from typing import List, Dict
import time

# argv
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="BlinkDL/rwkv-6-world")
parser.add_argument("--model-file", type=str, default="RWKV-x060-World-1B6-v2-20240208-ctx4096")
parser.add_argument("--no-instruct", action='store_true')
parser.add_argument("--no-use-system-prompt", action='store_true')
parser.add_argument("--max-tokens", type=int, default=256)

args = parser.parse_args(sys.argv[1:])

model_id = args.model_path
if model_id == None:
    exit
model_file = args.model_file
if model_file == None:
    exit

is_instruct = not args.no_instruct
use_system_prompt = not args.no_use_system_prompt
max_new_tokens = min(3500, args.max_tokens)

## Download the rwkv model
model_path = hf_hub_download(repo_id=model_id, filename=f"{model_file}.pth")
## Instantiate model from downloaded file
model = RWKV(model=model_path, strategy='cuda fp16')
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

DEFAULT_SYSTEM_PROMPT = "わたしは誠実で優秀な日本人のアシスタントです。"

# generations params
pipeline_args = PIPELINE_ARGS(
    temperature=1.0,
    top_p=0.85,
    top_k=0,
    alpha_frequency=0.2,
    alpha_presence=0.2,
    alpha_decay=0.996,
    token_ban=[],
    token_stop=[],
    chunk_len=256
)

#
def generate_chat_prompt(
    conversation: List[Dict[str, str]],
    add_generation_prompt=True,
) -> str:
    prompt = ""
    for message in conversation:
        role = message["role"]
        content = message["content"].strip().replace('\r\n','\n').replace('\n\n','\n')
        if message["role"] == "system":
            prompt += f"User: こんにちは\n\nAssistant: {content}\n\n"
        else:
            prompt += f"{role}: {content}\n\n"
    if add_generation_prompt:
        prompt += "Assistant:"
    return prompt

#
def generate_prompt(
    user_query: str,
    instruction: str=None,
    add_generation_prompt=True,
) -> str:
    prompt = ""
    prompt += f"Instruction: {instruction}\n\n"
    prompt += f"Input: {user_query}\n\n"
    if add_generation_prompt:
        prompt += f"Response:"
    return prompt

# callback function
def print_nolf(outstr):
    print(outstr, end="")

def q(
    user_query: str,
    history: List[Dict[str, str]]=None,
    instruction: str=None
) -> List[Dict[str, str]]:
    start = time.process_time()
    # messages
    messages = ""
    if is_instruct:
        messages = []
        if use_system_prompt:
            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            ]
        user_messages = [
            {"role": "User", "content": user_query}
        ]
    else:
        user_messages = user_query
    if history:
        user_messages = history + user_messages
    messages += user_messages
    # generation prompts
    if is_instruct:
        prompt = generate_chat_prompt(
            conversation=messages,
            add_generation_prompt=True,
        )
    else:
        prompt = generate_prompt(
            user_query=messages,
            instruction=instruction,
            add_generation_prompt=True,
        )
    print("--- prompt")
    print(prompt)
    print("--- output")
    # 推論
    output = pipeline.generate(
        ctx=prompt,
        token_count=max_new_tokens,
        args=pipeline_args,
        callback=print_nolf
    )
    if is_instruct:
        user_messages.append(
            {"role": "Assistant", "content": output}
        )
    else:
        user_messages += output
    end = time.process_time()
    ##
    input_ids = pipeline.encode(prompt)
    input_tokens = len(input_ids)
    output_ids = pipeline.encode(output)
    output_tokens = len(output_ids)
    total_time = end - start
    tps = output_tokens / total_time
    print("\n---")
    print(f"prompt tokens = {input_tokens:.7g}")
    print(f"output tokens = {output_tokens:.7g} ({tps:f} [tps])")
    print(f"   total time = {total_time:f} [s]")
    return user_messages

