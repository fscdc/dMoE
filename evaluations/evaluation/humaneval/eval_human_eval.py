"""
https://github.com/openai/simple-evals/blob/main/humaneval_eval.py

HumanEval: Evaluating Large Language Models Trained on Code
Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba 
https://arxiv.org/abs/2107.03374 https://github.com/openai/human-eval/ 
"""
import argparse

import json
import logging
import multiprocessing
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Tuple

# from human_eval.data import HUMAN_EVAL, read_problems
# from human_eval.evaluation import estimate_pass_at_k
# from human_eval.execution import check_correctness  # , unsafe_execute

from datasets import load_dataset

def time_evaluation(result_file):
    results = []
    f = open(result_file, 'r')
    for line in f:
        results.append(json.loads(line))
    f.close()

    total_time, total_token = 0, 0
    total = 0
    
    for idx, problem in enumerate(results):
        total += 1
        total_time += problem['time']
        total_token += problem['tokens']
        
    return {
        'total': total,
        'total_time': total_time,
        'total_token': total_token,
        'token/s': total_token / total_time,
    }

def find_code(completion):
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[0] if len(matches) >= 1 else completion
    extracted_answer = extracted_answer[
        extracted_answer.find(":\n    ") + 2 :
    ]  # remove signature
    return extracted_answer

# def evaluate_functional_correctness(
#     sample: dict[str, str],
#     completions: list[str],
#     n_workers: int = 4,
#     timeout: float = 3.0,
# ):
#     """
#     Evaluates the functional correctness of generated samples, and writes
#     results to f"{sample_file}_results.jsonl.gz"
#     """
#     import copy

#     # Check the generated samples against test suites.
#     with ThreadPoolExecutor(max_workers=n_workers) as executor:
#         futures = []
#         for i, completion in enumerate(completions):
#             args = (sample, completion, timeout, i)
#             future = executor.submit(check_correctness, *args)
#             futures.append(future)
#         results = []
#         for future in as_completed(futures):
#             result = future.result()
#             results.append(result)
#     passed = [int(r["passed"]) for r in results]
#     return passed

import evaluate as hf_evaluate
import os
import json
from .sanitize import sanitize

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
pass_at_k = hf_evaluate.load("code_eval")

def pass_at_1(references, predictions):
    return pass_at_k.compute(
        references=references,
        predictions=predictions,
        k=[1],
    )[0]["pass@1"]

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def evaluation(file_path):
    data = read_jsonl(file_path)

    # 提取 references 和 predictions
    references = [sample['task']['test']+f"\n\ncheck({sample['task']['entry_point']})" for sample in data]

    predictions = [
        [
            sanitize(
                sample['task']['prompt'] + "\n" + sample['answer'][0].split('```python\n', 1)[-1].split('```')[0], 
                sample['task']["entry_point"]
            )
        ]
        for sample in data
    ]

    pass_at_1s = [pass_at_1([reference], [prediction]) for reference, prediction in zip(references, predictions)]
    avg_pass_at_1 = sum(pass_at_1s) / len(pass_at_1s)


    total_time, total_token, total_useful_token, total_steps = 0, 0, 0, 0
    
    total = 0
    for sample in data:
        total += 1

        total_time += sample['time']
        total_token += sample['tokens']
        total_useful_token += sample['useful_tokens']

        total_steps += sample['steps']

    return {
        "Pass@1": avg_pass_at_1,
        'total': total,
        'total_time': total_time,
        'total_token': total_token,
        'avg_token': total_token / total,
        'avg_useful_token': total_useful_token / total,
        'token/s': total_token / total_time,
        'useful_token/s': total_useful_token / total_time,
        'avg_steps': total_steps / total,
    }


if __name__ == '__main__':
    evaluation('./result_summary/humaneval/origin_len_128_steps_128_low_confidence_block_128_cfg_0.0_cache_False_seed_42_revise_ratio_0.0_draft_gamma_0.6_all.jsonl')
    # args = argparse.ArgumentParser()
    # args.add_argument('--result_file', type=str, default='./result_summary/humaneval/origin_len_128_steps_128_low_confidence_block_128_cfg_0.0_cache_False_seed_42_revise_ratio_0.0_draft_gamma_0.6_all.json')
    # args.add_argument('--num-samples', type=int, default=None)
    # args = args.parse_args()
    
    # dataset = load_dataset('openai/openai_humaneval')['test']
    # results = json.load(open(args.result_file, 'r'))

    # total, s_correct = 0, 0
    # for idx, (answer, problem) in enumerate(zip(results,  dataset)):
    #     if args.num_samples is not None and idx >= args.num_samples:
    #         break
    #     print(answer['answer'])
        
    #     results = evaluate_functional_correctness(problem, find_code(answer['answer']))
    #     print(results)
    #     total = len(results)
    #     correct = sum(results)
    #     score = sum(results) / len(results)
    #     print(score)
    #     exit()

    #     total += 1
    #     s_correct += score

    # print(f"Strict Match Accuracy = {s_correct}/{total}  = {s_correct/total}")


