import os
import argparse

import pandas as pd
import re
from datasets import load_dataset

import json

from .grader import grade_answer

def extract_all_boxed_content(text):
    results = []
    start = 0

    while True:
        # Find the next occurrence of \boxed{
        start = text.find(r"\boxed{", start)
        if start == -1:
            break  # No more \boxed{ found

        brace_count = 0
        result = []
        i = start

        while i < len(text):
            char = text[i]
            result.append(char)

            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1

            # Stop when the braces are balanced
            if brace_count == 0 and result[-1] == '}':
                break

            i += 1

        # Append the matched content
        results.append(''.join(result))
        start = i + 1  # Move past the current match to find the next

    return results

def extract_boxed_text(solution):
    strict_prediction, soft_prediction = None, None
    prediction_match = extract_all_boxed_content(str(solution))
   
    if len(prediction_match) > 0:
        strict_prediction = prediction_match[-1]
        if strict_prediction is not None and '\\boxed' in strict_prediction:
            strict_prediction = strict_prediction.replace('\\boxed{', '')[:-1]
    else:
        patterns = [
            r"<answer>(.*?)</answer>",
            r"</answer>(.*?)</answer>",
            r"<answer>(.*?)<answer>",
            r"\*\*Answer:\*\* ([\d\.]+)",
            # last number 
            r"[-+]?\d*\.\d+|\d+",
        ]
        for pattern in patterns:
            prediction_match = re.findall(pattern, str(solution))
            if len(prediction_match) > 0:
                break
            
        if len(prediction_match) > 0:
            soft_prediction = prediction_match[-1]
        else:
            soft_prediction = None

    return strict_prediction, soft_prediction



def process_target(target, dataset):
    if dataset == 'gsm8k':
        target = int(target.split('#### ')[-1].replace(',',''))
        return target
    if dataset == 'math500':
        return target
    elif dataset == 'aime':
        return target 
    else:
        raise ValueError('Unknown dataset')
    
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

def evaluation(result_file):
    results = []
    f = open(result_file, 'r')
    for line in f:
        results.append(json.loads(line))
    f.close()

    total, s_correct, f_correct = 0, 0, 0
    total_time, total_token = 0, 0

    total_steps = 0
    total_experts = 0
    
    for idx, problem in enumerate(results):

        answer = problem['answer']
        strict_predict_ans, flexible_predict_ans = extract_boxed_text(answer)

        target = problem['task']['answer']
        target = process_target(target, 'math500')

        total += 1
        total_time += problem['time']
        total_token += problem['tokens']

        total_steps += problem['steps']
        total_experts += problem.get('unique_experts_count', 0)
        
        if strict_predict_ans is not None or flexible_predict_ans is not None: 
            try:
                if grade_answer(strict_predict_ans, target):
                    s_correct += 1
                    f_correct += 1
                elif grade_answer(flexible_predict_ans, target):
                    f_correct += 1
                       
            except:
                print('Error in extracting answers: ', strict_predict_ans, flexible_predict_ans)
                pass
        else:
            print('No answer found: ', idx, target)
            pass

    print(f"Strict Match Accuracy = {s_correct}/{total}  = {s_correct/total}")
    print(f"Soft Match Accuracy = {f_correct}/{total}  = {f_correct/total}")
    return {
        'strict_accuracy': s_correct / total,
        'soft_accuracy': f_correct / total,
        'strict_match': s_correct,
        'soft_match': f_correct,
        'total': total,
        'total_time': total_time,
        'total_token': total_token,
        'token/s': total_token / total_time,
        'avg_steps': total_steps / total if total > 0 else -1,
        'unique_experts': total_experts / total if total > 0 else -1,
    }

def passk_evaluation(result_files):
    """
    Pass@k evaluation for MATH500.
    A question is counted as correct if ANY of the result files answers it correctly.
    
    result_files: list[str], each file is a JSONL result file.
    """

    # Load all result files
    all_results = []
    for file in result_files:
        with open(file, "r") as f:
            results = [json.loads(line) for line in f]
            all_results.append(results)

    num_files = len(all_results)
    num_questions = len(all_results[0])

    strict_correct = 0
    soft_correct = 0

    for qid in range(num_questions):

        # Ground truth
        target = all_results[0][qid]['task']['answer']
        target = process_target(target, 'math500')

        strict_hit = False
        soft_hit = False

        # Check all result files for this question
        for fidx in range(num_files):

            answer = all_results[fidx][qid]['answer']
            strict_ans, flexible_ans = extract_boxed_text(answer)

            try:
                # strict match
                if grade_answer(strict_ans, target):
                    strict_hit = True
                    soft_hit = True
                    break

                # soft match
                if grade_answer(flexible_ans, target):
                    soft_hit = True
                    break

            except Exception as e:
                print(f"[Warning] Error grading answer in file {fidx}, qid {qid}: {e}")
                continue

        # Update match counters
        if strict_hit:
            strict_correct += 1
            soft_correct += 1
        elif soft_hit:
            soft_correct += 1

    # Final metrics
    return {
        "strict_accuracy": strict_correct / num_questions,
        "soft_accuracy": soft_correct / num_questions,
        "strict_match": strict_correct,
        "soft_match": soft_correct,
        "total": num_questions,
    }

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--result_file', type=str, default='results.csv')
    args.add_argument('--dataset', type=str, default='math500')
    args.add_argument('--split', type=str, default='test')
    args.add_argument('--num-samples', type=int, default=None)

    args.add_argument('--relax', action='store_true')
    args = args.parse_args()

    result_file = args.result_file    
    if args.dataset == 'gsm8k':
        dataset = load_dataset('openai/gsm8k', 'main')
        dataset_type = args.split
    elif args.dataset =='math500':
        dataset = load_dataset("HuggingFaceH4/MATH-500")
        dataset_type = 'test'
    elif args.dataset == 'aime':
        dataset = load_dataset('AI-MO/aimo-validation-aime')
        dataset_type = 'train'
    else:
        raise ValueError('Unknown dataset')
    
    results = json.load(open(result_file, 'r'))

    total, s_correct, f_correct = 0, 0, 0
    correct_length, wrong_length, all_length = [], [], []
    max_length, min_length = 0, 100000
    for idx, (ans, prob) in enumerate(zip(results,  dataset[dataset_type])):
        if args.num_samples is not None and idx >= args.num_samples:
            break
        
        if args.dataset == 'aime' and idx < 60:
            continue 

        strict_predict_ans, flexible_predict_ans = extract_boxed_text(ans['answer'])
        target = prob['answer']

        target = process_target(target, args.dataset)

        total += 1
        
        if strict_predict_ans is not None or flexible_predict_ans is not None: 
            try:
                correct_flag = False
                if grade_answer(strict_predict_ans, target):
                    s_correct += 1
                    f_correct += 1
                    correct_flag = True
                elif grade_answer(flexible_predict_ans, target):
                    f_correct += 1
                    correct_flag = True         

                #print(f"Correct = {correct_flag}, \tStrict Predict = {strict_predict_ans},\t Flexible Predict = {flexible_predict_ans}, \tTarget = {target}")
            except:
                #print('Error in extracting answers: ', strict_predict_ans, flexible_predict_ans)
                pass
        else:
            #print('No answer found: ', idx, target)
            pass

        #if total >= 224:
        #    break

    print(f"Strict Match Accuracy = {s_correct}/{total}  = {s_correct/total}")
    print(f"Soft Match Accuracy = {f_correct}/{total}  = {f_correct/total}")

