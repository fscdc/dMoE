import torch
from transformers import AutoTokenizer
from evaluations.models.modeling_llada2_moe_be_adaptive import LLaDA2MoeModelLM

MODEL_NAME = "FSCCS/dMoE-16B"

device = "cuda:0"

model = LLaDA2MoeModelLM.from_pretrained(
    MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16
).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

prompt = "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?" + "\nLet's think step by step\n"

messages = [[{"role": "user", "content": prompt}]]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

inputs = tokenizer(input_text, return_tensors="pt", padding_side="left")
input_ids = inputs["input_ids"].to(device)

with torch.no_grad():
    out, unique_experts_count = model.generate(
        input_ids,
        steps=32,
        gen_length=2048,
        block_length=32,
        temperature=0.0,
        eos_early_stop=True,
    )

generated = out[:, input_ids.shape[1]:]
result = tokenizer.batch_decode(generated, skip_special_tokens=True)

print("Output:", result[0])
print("Unique experts count:", unique_experts_count)
