import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer

import os
import time
import datetime
import argparse
from tqdm import tqdm

import json

import torch
import torch.distributed as dist

import time


def set_random_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dataloader_by_rank(ds, bsz, rank, world_size):
    index = 0
    while index * bsz * world_size < len(ds):
        start_idx = index * bsz * world_size + bsz * rank
        if start_idx >= len(ds):
            yield []
        elif start_idx + bsz >= len(ds):
            yield ds.select(range(start_idx, len(ds)))
        else:
            yield ds.select(range(start_idx,start_idx+bsz))

        index += 1



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg", type=float, default=0.0)

    parser.add_argument("--dataset", type=str, required=True)

    parser.add_argument("--bsz", type=int, default=8)
    parser.add_argument("--sampling-alg", type=str, default=None)
    parser.add_argument("--enable-cache", action="store_true", default=False)
    parser.add_argument("--cache-steps", type=int, default=2)

    parser.add_argument("--origin", action="store_true")
    parser.add_argument("--kv-cache", action="store_true")
    parser.add_argument("--q-cache", action="store_true")
    parser.add_argument("--block-cache", action="store_true")


    # Hyper-parameter for q-cache
    parser.add_argument("--window-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation-only", action="store_true", default=False)
    parser.add_argument("--time-evaluation", action="store_true", default=False)
    parser.add_argument("--subset", action="store_true", default=False)


    # @sicheng: Hyper-parameter for draft decoding
    parser.add_argument("--enable-draft-decoding", action="store_true", default=False)
    parser.add_argument("--draft-gamma", type=float, default=0.0)

    # @sicheng: Hyper-parameter for revising
    parser.add_argument("--enable-revise", action="store_true", default=False)
    parser.add_argument("--revise-ratio", type=float, default=0.0)
    parser.add_argument("--revise-draft-gamma", type=float, default=0.0)

    parser.add_argument("--select-mode", type=str, default="random")
    parser.add_argument("--select-score-thresh", type=float, default=0.5)

    
    # @sicheng: Hyper-parameter for efficient test-time scaling
    parser.add_argument("--enable-adaptive-decoding", action="store_true", default=False)
    parser.add_argument("--entropy-thresh", type=float, default=0.5)
    parser.add_argument("--enable-remask-sampling", action="store_true", default=False)
    parser.add_argument("--sampling-upper-bound", type=int, default=5)
    parser.add_argument("--model-name", type=str, default="/local_home1/fengsicheng/pretrained_models/LLaDA2.0-mini") # 



    parser.add_argument("--enable-record", action="store_true", default=False)
    parser.add_argument("--timer-record", action="store_true", default=False)


    # add more for moe model
    parser.add_argument("--eos-early-stop", action="store_true", default=False)
    parser.add_argument("--des", action="store_true", default=False)
    parser.add_argument("--origin-21", action="store_true", default=False)

    parser.add_argument("--be", action="store_true", default=False)
    parser.add_argument("--be-group", action="store_true", default=False)
    parser.add_argument("--be-adaptive", action="store_true", default=False)

    args = parser.parse_args()

    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=36000))
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    device = 'cuda:{}'.format(rank)

    if args.origin:
        from models.modeling_llada2_moe import LLaDA2MoeModelLM
    elif args.des:
        from models.modeling_llada2_moe_des import LLaDA2MoeModelLM
    elif args.be:
        from models.modeling_llada2_moe_be import LLaDA2MoeModelLM
    elif args.be_group:
        from models.modeling_llada2_moe_be_group import LLaDA2MoeModelLM
    elif args.be_adaptive:
        from models.modeling_llada2_moe_be_adaptive import LLaDA2MoeModelLM
    elif args.origin_21:
        from models.modeling_llada21_moe import LLaDA2MoeModelLM
    else:
        raise NotImplementedError

    model = LLaDA2MoeModelLM.from_pretrained(
        args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, #device_map=device
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    from datasets import load_dataset
    if args.dataset == 'gsm8k':
        from evaluation.gsm8k.prompt import format_prompt
        from evaluation.gsm8k.eval_gsm8k import evaluation, time_evaluation
        ds = load_dataset('openai/gsm8k', 'main')['test'] # test
    elif args.dataset == 'math500':
        from evaluation.math500.prompt import format_prompt
        from evaluation.math500.eval_math500 import evaluation, time_evaluation
        ds = load_dataset("HuggingFaceH4/MATH-500")['test']
    elif args.dataset == 'gpqa': # Use simple-eval
        from evaluation.gpqa.prompt import format_prompt
        from evaluation.gpqa.eval_gpqa import evaluation, time_evaluation
        ds = load_dataset("Idavidrein/gpqa", "gpqa_main")['train']
    elif args.dataset == 'humaneval':
        from evaluation.humaneval.prompt import format_prompt
        from evaluation.humaneval.eval_human_eval import evaluation, time_evaluation
        ds = load_dataset('openai/openai_humaneval')['test'] # test
    elif args.dataset == 'mmlu':
        from evaluation.mmlu.prompt import format_prompt
        from evaluation.mmlu.eval_mmlu import evaluation, time_evaluation
        ds = load_dataset('cais/mmlu', 'high_school_mathematics')['test'] # high_school_mathematics
    elif args.dataset == 'mbpp':
        from evaluation.mbpp.prompt import format_prompt
        from evaluation.mbpp.eval_mbpp import evaluation, time_evaluation
        ds = load_dataset('google-research-datasets/mbpp')['test']
    elif args.dataset == 'deepscaler':
        from evaluation.deepscaler.prompt import format_prompt
        from evaluation.deepscaler.eval_deepscaler import evaluation, time_evaluation
        ds = load_dataset('agentica-org/DeepScaleR-Preview-Dataset')['train']
    elif args.dataset == 'arc-c':
        from evaluation.arc.prompt import format_prompt
        from evaluation.arc.eval_arc import evaluation, time_evaluation
        ds = load_dataset('allenai/ai2_arc', 'ARC-Challenge')['test']
    else:
        raise NotImplementedError
    results = []

    model = model.to(rank)
    model.eval()

    # a = torch.zeros(90000, 90000).to(rank)

    set_random_seed(args.seed)

    if args.time_evaluation:
        if args.subset:
            root_dir = 'llada_speed_subset'
        else:
            root_dir = 'llada_final'
    else:
        root_dir = 'result_summary'

    if args.model_name == 'GSAI-ML/LLaDA-1.5':
        root_dir += '_llada1.5'
    elif "LLaDA2.0-mini" in args.model_name:
        root_dir += '_llada2.0-mini'
    elif "LLaDA2.1-mini" in args.model_name:
        root_dir += '_llada2.1-mini'
    elif "LLaDA-MoE" in args.model_name:
        root_dir += '_llada_moe'

    if args.des:
        root_dir += '_des'
    elif args.be:
        root_dir += '_be'
    elif args.be_group: 
        root_dir += '_be_group'
    elif args.be_adaptive:
        root_dir += '_be_adaptive'
    elif args.timer_record:
        root_dir += '_timer'

    if args.subset:
        root_dir += '_subset'

    if args.enable_record:
        root_dir += '_record_plus'

    if args.enable_revise:
        root_dir += '_revise'
    elif args.enable_adaptive_decoding:
        root_dir += '_prepare'
    else:
        pass

    if args.enable_remask_sampling:
        root_dir = root_dir.replace('prepare', 'remask_sampling')

    dir_name = f'{root_dir}/{args.dataset}'
    os.makedirs(dir_name+'/subprocess', exist_ok=True)
    if not args.enable_cache:
        # @sicheng: for draft decoding and revise
        # filename = f"{dir_name}/subprocess/origin_len_{args.seq_len}_steps_{args.steps}_{args.sampling_alg}_block_{args.block_size}_cfg_{args.cfg}_cache_{args.enable_cache}_seed_{args.seed}_revise_ratio_{args.revise_ratio}_draft_gamma_{args.draft_gamma}_revise_gamma_{args.revise_draft_gamma}_select_mode_{args.select_mode}_score_thresh_{args.select_score_thresh}"
        
        # @sicheng: for adaptive test-time scaling
        filename = f"{dir_name}/subprocess/origin_len_{args.seq_len}_steps_{args.steps}_{args.sampling_alg}_block_{args.block_size}_cfg_{args.cfg}_cache_{args.enable_cache}_seed_{args.seed}_temp_{args.temperature}"
    elif args.kv_cache:
        filename = f"{dir_name}/subprocess/kv_cache_decode_len_{args.seq_len}_steps_{args.steps}_{args.sampling_alg}_block_{args.block_size}_cfg_{args.cfg}_cache_{args.cache_steps}_seed_{args.seed}"
    elif args.q_cache:
        filename = f"{dir_name}/subprocess/q_cache_mask_window_{args.window_size}_len_{args.seq_len}_steps_{args.steps}_{args.sampling_alg}_block_{args.block_size}_cfg_{args.cfg}_cache_{args.cache_steps}_seed_{args.seed}"
    elif args.block_cache:
        filename = f"{dir_name}/subprocess/block_cache_window_{args.window_size}_len_{args.seq_len}_steps_{args.steps}_{args.sampling_alg}_block_{args.block_size}_cfg_{args.cfg}_cache_{args.cache_steps}_seed_{args.seed}"
    else:
        raise NotImplementedError

    if args.enable_revise:
        filename += '_revise'

    if not args.evaluation_only:
        f = open(f'{filename}_rank_{rank}.json', 'w') 

    if args.subset:
        ds = ds.select(range(128))
        # ds = ds.select(range(147, 156))
    
    ds = ds.add_column("index", list(range(len(ds))))

    for sample in tqdm(dataloader_by_rank(ds, args.bsz, rank, world_size), total=len(ds) // (world_size * args.bsz)+1):
        
        if args.evaluation_only:
            break

        m = []
        if len(sample) == 0:
            break

        for p in sample:
            question, answer = format_prompt(p)
            m.append(
                [{
                    "role": "user", 
                    "content": question
                }] 
            )

        prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

        bsz = len(m)

        input_prompt = tokenizer(
            prompt,
            padding_side = 'left', padding = 'longest',
            return_tensors = 'pt'
        )
        input_ids = input_prompt['input_ids'].to(rank)#.unsqueeze(0)#.repeat(bsz, 1)
        attention_mask = input_prompt['attention_mask'].to(rank)

        start_time = time.time()

        out = None
        if args.enable_draft_decoding:
            pass
        elif args.enable_remask_sampling:
            # @sicheng: rebuttal on llada 2.1 remask sampling
            out = model.tts_generate(input_ids, steps=args.steps,
                gen_length=args.seq_len, 
                block_length=args.block_size,
                temperature=args.temperature,
                eos_early_stop=args.eos_early_stop,
                sampling_upper_bound=args.sampling_upper_bound,
                tokenizer=tokenizer,
                dataset=args.dataset,)  
        else:
            if args.enable_record:
                pass
            elif args.timer_record:
                out, unique_experts_count, timer_records = model.generate_timer(input_ids, steps=args.steps,
                    gen_length=args.seq_len, 
                    block_length=args.block_size,
                    temperature=args.temperature,
                    eos_early_stop=args.eos_early_stop,)
            else:
                # llada 2.0
                out, unique_experts_count = model.generate(input_ids, steps=args.steps,
                    gen_length=args.seq_len, 
                    block_length=args.block_size,
                    temperature=args.temperature,
                    eos_early_stop=args.eos_early_stop,)         

        
        end_time = time.time()
        out = out[:, input_ids.shape[1]:].contiguous()
        
        result = tokenizer.batch_decode(out, skip_special_tokens=True)

        for i, p in enumerate(sample): # 156892 is eos token id
            useful_token = (out[i] == 156892).nonzero(as_tuple=True)[0].sort().values
            if len(useful_token) > 0:
                useful_token = useful_token[0].item()
            else:
                useful_token = out.shape[1]

            idx = p['index']

            res = {
                    "answer": result,
                    "task": p,
                    "time": (end_time - start_time) / len(sample),
                    "tokens": out.shape[1],
                    "useful_tokens": useful_token,
                    "answer_index": answer,
                    "steps": steps if args.enable_draft_decoding or args.enable_adaptive_decoding else -1,
                    "unique_experts_count": unique_experts_count if not args.enable_draft_decoding and not args.enable_adaptive_decoding else None,
                    "timer_records": timer_records if args.timer_record else None,
                }


            f.write(json.dumps(res) + '\n')
            f.flush()
    
    if not args.evaluation_only:
        f.close()
    dist.barrier()

    if rank == 0:
        name = filename.split('/')[-1]
        all_f_name = f"{dir_name}/{name}" + '_all.json'
        if args.dataset == 'humaneval':
            all_f_name = f"{dir_name}/{name}" + '_all.jsonl'
        all_f = open(all_f_name, 'w')
        for i in range(world_size):
            f = open(filename + f'_rank_{i}.json', 'r')
            for line in f:
                all_f.write(line)
            f.close()

        all_f.close()
        
        if args.time_evaluation:
            final_result = time_evaluation(all_f_name)
            final_result['config'] = args.__dict__
            result_path = f"{dir_name}/time_{name}.json"
        else:
            final_result = evaluation(all_f_name)
            final_result['config'] = args.__dict__
            result_path = f"{dir_name}/result_{name}.json"

        with open(result_path + '.json', 'w') as f:
            json.dump(final_result, f, indent=4)     

        print(final_result)


if __name__ == '__main__':
    main()