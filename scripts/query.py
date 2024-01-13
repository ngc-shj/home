import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from typing import List, Dict
import time

if len(sys.argv) < 2:
    exit()

model_id = sys.argv[1]

# トークナイザーとモデルの準備
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"

# generation params
max_new_tokens = 256
generation_params = {
    "do_sample": True,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 40,
    "max_new_tokens": max_new_tokens,
    "repetition_penalty": 1.1
}

def q(
    user_query: str,
    chat_history: List[Dict[str, str]]=None
):
    start = time.process_time()
    # messages
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
    ]
    user_messages = [
        {"role": "user", "content": user_query}
    ]
    if chat_history:
        user_messages = chat_history + user_messages
    messages += user_messages
    # generateion prompts
    prompt = tokenizer.apply_chat_template(
        conversation=messages,
        add_generation_prompt=True,
        tokenize=False
    )
    print("--- prompt")
    print(prompt)
    print("---")
    #
    input_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
        return_tensors="pt"
    )
    # 推論
    output_ids = model.generate(
        input_ids.to(model.device),
        streamer=streamer,
        **generation_params
    )
    output = tokenizer.decode(
        output_ids[0][input_ids.size(1) :],
        skip_special_tokens=True
    )
    user_messages.append(
        {"role": "assistant", "content": output}
    )
    end = time.process_time()
    ##
    input_tokens = len(input_ids[0])
    output_tokens = len(output_ids[0][input_ids.size(1) :])
    total_time = end - start
    tps = output_tokens / total_time
    print(f"prompt tokens = {input_tokens:.7g}")
    print(f"output tokens = {output_tokens:.7g} ({tps:f} [tps])")
    print(f"   total time = {total_time:f} [s]")
    return user_messages

chat_history = ""
chat_history = q("ドラえもんとはなにか")
chat_history = q("続きを教えてください", chat_history)

