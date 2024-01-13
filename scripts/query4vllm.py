import sys
from vllm import LLM, SamplingParams
from typing import List, Dict
import time

if len(sys.argv) < 2:
    exit()

model_id = sys.argv[1]

# トークナイザーとモデルの準備
model = LLM(
    model=model_id,
    dtype="auto",
    trust_remote_code=True,
    #tensor_parallel_size=2,
    #max_model_len=1024
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
    input_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )
    # 推論
    outputs = model.generate(
        sampling_params=generation_params,
        prompt_token_ids=[input_ids],
    )
    print(outputs)
    output = outputs[0]
    print("--- prompt")
    print(output.prompt)
    print(prompt)
    print("--- output")
    print(output.outputs[0].text)
    user_messages.append(
        {"role": "assistant", "content": output.outputs[0].text}
    )
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

chat_history = ""
chat_history = q("ドラえもんとはなにか")
chat_history = q("続きを教えてください", chat_history)

