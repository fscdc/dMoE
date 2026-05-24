import base64
import time
from typing import Any, Callable, Optional
import os

from ..types import MessageList, SamplerBase

import torch
import gc
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

class LLaMASampler(SamplerBase):
    """
    Sample from LLaDA
    """
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        generate_fn: Optional[Callable] = None,
        generation_kwargs: Optional[dict] = None,
        kv_cache_masked: Optional[bool] = False,
        kv_cache_decoded: Optional[bool] = False,
    ):  
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,                    
        ).eval().requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.generation_kwargs = generation_kwargs
    
    def init_model(self):
        self.model = self.model.cuda()

    def _handle_text(self, text: str) -> dict[str, Any]:
        return {"type": "input_text", "text": text}

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": role, "content": content}

    def _free_memory(self):
        del self.model

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        import time
        time.sleep(10)

    def __call__(self, seq_idx: int, message_list: MessageList) -> str:

        with torch.inference_mode(): 
            full_message = [
                {'role': 'system', 'content': 'You are a helpful AI assistant.'}
            ] + message_list
            prompt = self.tokenizer.apply_chat_template(full_message, add_generation_prompt=True, tokenize=False)
            #print(prompt)

            model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
            generated_ids = self.model.generate(
                **model_inputs,
                **self.generation_kwargs
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            length = len(generated_ids[0])
            res = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            #print(response)
       
        return seq_idx, res
