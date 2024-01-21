import sys
import argparse
from vllm import LLM, SamplingParams
from typing import List, Dict
import time

# argv
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default=None)
parser.add_argument("--no-instruct", action='store_true')
parser.add_argument("--no-use-system-prompt", action='store_true')

args = parser.parse_args(sys.argv[1:])

model_id = args.model_path
if model_id == None:
    exit

is_instruct = not args.no_instruct
use_system_prompt = not args.no_use_system_prompt

# トークナイザーとモデルの準備
model = LLM(
    model=model_id,
    dtype="auto",
    trust_remote_code=True,
    #tensor_parallel_size=2,
    #max_model_len=1024,
    #quantization="awq",
    #gpu_memory_utilization=0.2
)
tokenizer = model.get_tokenizer()

DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"

# generation params
max_new_tokens=1024
generation_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    top_k=40,
    max_tokens=max_new_tokens,
    repetition_penalty=1.1
)

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
    input_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )
    print("--- prompt")
    print(prompt)
    print("--- output")
    # 推論
    outputs = model.generate(
        sampling_params=generation_params,
        prompt_token_ids=[input_ids],
    )
    print(outputs)
    output = outputs[0]
    print(output.outputs[0].text)
    if is_instruct:
        user_messages.append(
            {"role": "assistant", "content": output.outputs[0].text}
        )
    else:
        user_messages += output.outputs[0].text
    end = time.process_time()
    ##
    input_tokens = len(output.prompt_token_ids)
    output_tokens = len(output.outputs[0].token_ids)
    total_time = end - start
    tps = output_tokens / total_time
    print(f"prompt tokens = {input_tokens:.7g}")
    print(f"output tokens = {output_tokens:.7g} ({tps:f} [tps])")
    print(f"   total time = {total_time:f} [s]")
    return user_messages

print('history = ""')
print('history = q("ドラえもんとはなにか")')
print('history = q("続きを教えてください", history)')

