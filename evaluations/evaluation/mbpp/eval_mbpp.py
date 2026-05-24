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

from .human_eval.evaluation import evaluate_functional_correctness

from datasets import load_dataset

def find_code(completion):
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[0] if len(matches) >= 1 else completion
    #extracted_answer = extracted_answer[
    #    extracted_answer.find(":\n    ") + 2 :
    #]  # remove signature
    return extracted_answer.split('# Test cases')[0].strip()  # remove test cases

def evaluation(result_file):

    total_time, total_token, total_steps = 0, 0, 0

    total_experts = 0

    with open('tmp_mbpp.json', 'w') as f:
        old_f = open(result_file, 'r')
        for line in old_f:
            res = json.loads(line)
            f.write(json.dumps({
                'generation': find_code(res['answer'][0]),
                'task_id': res['task']['task_id'],
            }) + '\n')
            total_time += res['time']
            total_token += res['tokens']
            total_steps += res['steps']
            total_experts += res['unique_experts_count']
            f.flush()
        old_f.close()

    res = evaluate_functional_correctness(
        input_file='tmp_mbpp.json',
        problem_file='data/mbpp_test.jsonl',
        tmp_dir="llada_final/tmp",
        timeout=10,
        language="python"
    )
    res['total_time'] = total_time
    res['total_token'] = total_token
    res['token/s'] = total_token / total_time,
    res['total_steps'] = total_steps
    res['avg_steps'] = total_steps / res['total']
    res['unique_experts'] = total_experts / res['total'] if res['total'] > 0 else 0

    return res

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

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--result_file', type=str, default='results.csv')
    args.add_argument('--num-samples', type=int, default=None)
    args = args.parse_args()
    
    dataset = load_dataset('google-research-datasets/mbpp')['test']
    results = json.load(open(args.result_file, 'r'))

    
    with open('tmp_mbpp.json', 'w') as f:
        for idx, (answer, problem) in enumerate(zip(results,  dataset)):
            if args.num_samples is not None and idx >= args.num_samples:
                break
            res = {
                "generation": find_code(answer['answer']),
                "task_id": problem['task_id'],
            }
            f.write(json.dumps(res) + '\n')
            f.flush()

    res = evaluate_functional_correctness(
        input_file='tmp_mbpp.json',
        problem_file='data/mbpp_test.jsonl',
        tmp_dir="results_final_2/mbpp_tmp",
        timeout=10,
        language="python"
    )
    print(res)
        