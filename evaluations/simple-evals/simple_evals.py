import json
import argparse
import pandas as pd
from . import common
from .browsecomp_eval import BrowseCompEval
from .drop_eval import DropEval
from .gpqa_eval import GPQAEval
from .humaneval_eval import HumanEval
from .math_eval import MathEval
from .mgsm_eval import MGSMEval
from .mmlu_eval import MMLUEval
from .simpleqa_eval import SimpleQAEval
from .sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
)
from .sampler.o_chat_completion_sampler import OChatCompletionSampler
from .sampler.responses_sampler import ResponsesSampler
from .sampler.claude_sampler import ClaudeCompletionSampler, CLAUDE_SYSTEM_MESSAGE_LMSYS

from .sampler.opensource_sampler import LLaDASampler
from .sampler.llama_sampler import LLaMASampler

import torch
from torch.distributed import scatter_object_list, gather_object
import torch.distributed as dist

def init_distributed():
    from datetime import datetime, timedelta
    dist.init_process_group(
        backend="nccl", timeout=timedelta(hours=24)
    )
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world

def main():
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using different samplers and evaluations."
    )

    parser.add_argument(
        "--task", type=str, required=True, choices=["mmlu", "math", "gpqa", "humaneval"]
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument("--model", type=str, help="Select a model by name")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )
    # Generation Config
    parser.add_argument(
        "--remasking", type=str, default="low_confidence", help="Remasking strategy"
    )
    parser.add_argument(
        "--steps", type=int, default=128, help="Number of sampling steps"
    )
    parser.add_argument(
        "--max-len", type=int, default=128, help="Length of generated sequence"
    )
    parser.add_argument(
        "--block", type=int, default=64, help="Block length for sampling"
    )

    # for Cache
    parser.add_argument("--kv-cache-masked", action="store_true")
    parser.add_argument("--kv-cache-decoded", action="store_true")
    
    parser.add_argument("--cache-steps", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=0)

    args = parser.parse_args()

    if torch.cuda.device_count() > 1:
        rank, world = init_distributed()
    else:
        world, rank = 1, 0

    models = {
        "llada": LLaDASampler(
            model_name="GSAI-ML/LLaDA-8B-Instruct",
            generation_kwargs={
                'steps': args.steps, 
                'gen_length': args.max_len, 
                'block_length': args.block, 
                'temperature': 0., 
                'cfg_scale': 0., 
                'remasking': args.remasking,
                'enable_cache': args.kv_cache_masked or args.kv_cache_decoded,
                'cache_reloading_step': args.cache_steps,
                'window_size': args.window_size
            },
            kv_cache_masked=args.kv_cache_masked,
            kv_cache_decoded=args.kv_cache_decoded
        ),
        "llama": LLaMASampler(
            model_name = "meta-llama/Llama-3.1-8B-Instruct",
            generation_kwargs={
                'max_new_tokens': args.max_len
            }
        )
    }

    if args.list_models:
        print("Available models:")
        for model_name in models.keys():
            print(f" - {model_name}")
        return

    if args.model:
        if args.model not in models:
            print(f"Error: Model '{args.model}' not found.")
            return
        models = {args.model: models[args.model]}

    grading_sampler = ChatCompletionSampler(model="gpt-4o")
    equality_checker = ChatCompletionSampler(model="gpt-4-turbo-preview")
    # ^^^ used for fuzzy matching, just for math

    def get_evals(eval_name, debug_mode):
        num_examples = (
            args.examples if args.examples is not None else (5 if debug_mode else None)
        )
        # Set num_examples = None to reproduce full evals
        match eval_name:
            case "mmlu":
                return MMLUEval(num_examples=10 if debug_mode else num_examples)
            case "math":
                return MathEval(
                    equality_checker=equality_checker,
                    num_examples=num_examples,
                    n_repeats=1 if debug_mode else 10,
                )
            case "gpqa":
                return GPQAEval(
                    n_repeats=1 if debug_mode else 1, num_examples=num_examples,
                )
            case "mgsm":
                return MGSMEval(num_examples_per_lang=10 if debug_mode else 250)
            case "drop":
                return DropEval(
                    num_examples=10 if debug_mode else num_examples,
                    train_samples_per_prompt=3,
                )
            case "humaneval":
                return HumanEval(num_examples=10 if debug_mode else num_examples)
            case "simpleqa":
                return SimpleQAEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                )
            case "browsecomp":
                return BrowseCompEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                )
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    evals = {
        eval_name: get_evals(eval_name, args.debug)
        for eval_name in [args.task]
    }
    debug_suffix = "_DEBUG" if args.debug else ""
    print(debug_suffix)
    mergekey2resultpath = {}
    for model_name, sampler in models.items():
        sampler.init_model()
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler, rank = rank, world = world)

            # ^^^ Gather from different gpus:
            sampler._free_memory()
            print("Start aggregate results")
            if world > 1:
                gathered = [None] * world
                dist.all_gather_object(gathered, result)   
                flat = [o for sub in gathered for o in sub]  # 展平
                result = flat

            result = common.aggregate_results(result)

            # ^^^ how to use a sampler
            if (world > 1 and rank ==0) or world == 1:
                file_stem = f"{eval_name}_{model_name}"
                file_stem += f"_{args.remasking}_steps_{args.steps}_len_{args.max_len}_block_{args.block}"
                file_stem += f"_decoded_{args.kv_cache_decoded}_masked_{args.kv_cache_masked}_cache{args.cache_steps}_window{args.window_size}"
                report_filename = f"simple_eval_results/{file_stem}{debug_suffix}.html"
                print(f"Writing report to {report_filename}")
                with open(report_filename, "w") as fh:
                    fh.write(common.make_report(result))
                metrics = result.metrics | {"score": result.score}
                print(metrics)
                result_filename = f"simple_eval_results/{file_stem}{debug_suffix}.json"
                with open(result_filename, "w") as f:
                    f.write(json.dumps(metrics, indent=2))
                print(f"Writing results to {result_filename}")
                mergekey2resultpath[f"{file_stem}"] = result_filename

            if world > 1:
                dist.barrier()
    
    if (world > 1 and rank ==0) or world == 1:
        merge_metrics = []
        for eval_model_name, result_filename in mergekey2resultpath.items():
            try:
                result = json.load(open(result_filename, "r+"))
            except Exception as e:
                print(e, result_filename)
                continue
            result = result.get("f1_score", result.get("score", None))
            eval_name = eval_model_name[: eval_model_name.find("_")]
            model_name = eval_model_name[eval_model_name.find("_") + 1 :]
            merge_metrics.append(
                {"eval_name": eval_name, "model_name": model_name, "metric": result}
            )
        merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
            index=["model_name"], columns="eval_name"
        )
        print("\nAll results: ")
        print(merge_metrics_df.to_markdown())
        return merge_metrics


if __name__ == "__main__":
    main()
