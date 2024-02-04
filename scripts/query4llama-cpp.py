import sys
import argparse
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from transformers import AutoTokenizer
from typing import List, Dict
import time

# argv
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default=None)
parser.add_argument("--ggml-model-path", type=str, default=None)
parser.add_argument("--ggml-model-file", type=str, default=None)
parser.add_argument("--no-instruct", action='store_true')
parser.add_argument("--no-use-system-prompt", action='store_true')
parser.add_argument("--max-tokens", type=int, default=256)
parser.add_argument("--n-ctx", type=int, default=2048)
parser.add_argument("--n-threads", type=int, default=1)
parser.add_argument("--n-gpu-layers", type=int, default=-1)

args = parser.parse_args(sys.argv[1:])

## check and set args
model_id = args.model_path
if model_id == None:
    exit
if args.ggml_model_path == None:
    exit
if args.ggml_model_file == None:
    exit

is_instruct = not args.no_instruct
use_system_prompt = not args.no_use_system_prompt
max_new_tokens = args.max_tokens
n_ctx = args.n_ctx
n_threads = args.n_threads
n_gpu_layers = args.n_gpu_layers

## Instantiate tokenizer from base model
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)

## Download the GGUF model
ggml_model_path = hf_hub_download(
    args.ggml_model_path,
    filename=args.ggml_model_file
)

## Instantiate model from downloaded file
model = Llama(
    model_path=ggml_model_path,
    n_ctx=n_ctx,
    n_threads=n_threads,
    n_gpu_layers=n_gpu_layers
)

DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"

# generation params
# https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/llama.py#L1268
generation_params = {
    #"do_sample": True,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 40,
    "max_tokens": max_new_tokens,
    "repeat_penalty": 1.1,
}


def q(
    user_query: str,
    history: List[Dict[str, str]]=None
):
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
            {"role": "user", "content": user_query}
        ]
    else:
        user_messages = user_query
    if history:
        user_messages = history + user_messages
    messages += user_messages
    # generation prompts
    if is_instruct:
        prompt = tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=False
        )
    else:
        prompt = messages
    print("--- prompt")
    print(prompt)
    print("--- output")
    # 推論
    outputs = model.create_completion(
        prompt=prompt,
        #echo=True,
        #stream=True,
        **generation_params
    )
    #for output in outputs:
    #    print(output["choices"][0]["text"], end='')
    output = outputs["choices"][0]["text"]
    print(output)
    if is_instruct:
        user_messages.append(
            {"role": "assistant", "content": output}
        )
    else:
        user_messages += output
    end = time.process_time()
    ##
    input_tokens = outputs["usage"]["prompt_tokens"]
    output_tokens = outputs["usage"]["completion_tokens"]
    total_time = end - start
    tps = output_tokens / total_time
    print(f"prompt tokens = {input_tokens:.7g}")
    print(f"output tokens = {output_tokens:.7g} ({tps:f} [tps])")
    print(f"   total time = {total_time:f} [s]")
    return user_messages

print('history = ""')
print('history = q("ドラえもんとはなにか")')
print('history = q("続きを教えてください", history)')

