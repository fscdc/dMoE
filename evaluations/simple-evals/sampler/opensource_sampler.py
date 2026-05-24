import base64
import time
from typing import Any, Callable, Optional
import os

from ..types import MessageList, SamplerBase

import torch
import gc
from transformers import AutoModel, AutoTokenizer

class LLaDASampler(SamplerBase):
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
        if not kv_cache_masked and not kv_cache_decoded:
            from generation_utils.llada_generate import generate as llada_ori_generate
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,                    
            ).eval().requires_grad_(False)
            self.generate_fn = llada_ori_generate
        elif kv_cache_decoded:
            from models.modeling_llada_kv_cache import LLaDAModelLM
            from generation_utils.kv_cache import generate as llada_kv_generate
            self.model = LLaDAModelLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,                    
            ).eval().requires_grad_(False)
            self.generate_fn = llada_kv_generate
        elif kv_cache_masked:
            from models.modeling_llada_qcache_improved import LLaDAModelLM
            from generation_utils.q_cache_improved import generate as llada_kv_generate_masked
            self.model = LLaDAModelLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,                    
            ).eval().requires_grad_(False)
            self.generate_fn = llada_kv_generate_masked
        else:
            raise NotImplementedError()

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
            prompt = self.tokenizer.apply_chat_template(message_list, add_generation_prompt=True, tokenize=False)

            #print(prompt)
            #print(message_list)
            input_ids = self.tokenizer(
                [prompt],
                padding_side = 'left',
                padding = 'longest'
            )['input_ids']
            input_ids = torch.tensor(input_ids).to(self.model.device)

            #set_random_seed(42)

            out = self.generate_fn(
                self.model, self.tokenizer, input_ids, 
                **self.generation_kwargs
                #steps=128, gen_length=128, block_length=64, 
                #temperature=0., cfg_scale=0., 
                #remasking='random',
                #enable_cache=True,
                #cache_reloading_step=4,
                #window_size=args.window_size
            )

            res = self.tokenizer.batch_decode(
                out[:, input_ids.shape[1]:], 
                skip_special_tokens=True
            )[0]
            #print(prompt)
            #print(res)
       
        return seq_idx, res
