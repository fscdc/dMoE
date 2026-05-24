import os
import argparse

import pandas as pd
import re
from datasets import load_dataset

import json

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

def draw_length_distribution(length, correct_length, wrong_length, file_name):
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter
    # Adjusting the bin size for grouping
    bin_size = 10

    # Grouping values into bins for both arrays
    bins_array0 = [min(val, 1000) // bin_size * bin_size for val in length]
    bins_array1 = [min(val, 1000) // bin_size * bin_size for val in correct_length]
    bins_array2 = [min(val, 1000) // bin_size * bin_size for val in wrong_length]

    # Calculate frequencies for binned data
    freq_bins_array0 = Counter(bins_array0)
    freq_bins_array1 = Counter(bins_array1)
    freq_bins_array2 = Counter(bins_array2)

    # Sorting data for consistent plotting
    binned_values0, binned_freqs0 = zip(*sorted(freq_bins_array0.items()))
    binned_values1, binned_freqs1 = zip(*sorted(freq_bins_array1.items()))
    binned_values2, binned_freqs2 = zip(*sorted(freq_bins_array2.items()))

    # Plotting
    plt.figure(figsize=(12, 6))

    plt.bar(binned_values0, binned_freqs0, width=bin_size * 0.8, label='All', alpha=0.7, align='center')
    plt.bar(binned_values1, binned_freqs1, width=bin_size * 0.8, label='Correct', alpha=0.7, align='center')
    plt.bar(binned_values2, binned_freqs2, width=bin_size * 0.8, label='Incorrect', alpha=0.7, align='center')

    # Adding labels and legend
    plt.xlabel('Token Length', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Frequency Distribution with Bin Size {bin_size}', fontsize=14)
    plt.xticks(ticks=np.arange(0, max(max(binned_values1), max(binned_values2)) + bin_size, bin_size))
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Display the plot
    plt.savefig(file_name)


def extract_boxed_text(document):
    # Match content inside \boxed{...}
    pattern = r"\\boxed\{([^}]*)\}"
    matches = re.findall(pattern, document)

    # Extract fallback numbers (just keep your logic)
    numbers = re.findall(r'\b\d+\.?\d*\b', document)
    f_matches = [numbers[-1]] if numbers else None

    if len(matches) == 1:
        # DIRECT FIX: keep letters instead of filtering by digits
        content = matches[-1].strip()
        s_matches = [content]  # return as list to mimic original style
        f_matches = s_matches  # overwrite fallback with the boxed content
    elif len(matches) > 1:
        content = matches[-1].strip()
        s_matches = [content]
    else:
        s_matches = matches  # empty list

    return s_matches, f_matches



def extract_boxed_text_most_relaxed(document):
    # Regular expression to find text within \boxed{}
    pattern = r"\\boxed\{([^}]*)\}"
    #pattern = r'#### \s*(.*)'
    matches = re.findall(pattern, document)
    #if len(matches) != 1: 
        #print(document)
        #print("Flexible Matching: ")
    numbers = re.findall(r'\b\d+[\.\,]?\d*\b', document)
    #print(numbers[-2:])
    if len(numbers) >= 2:
        f_matches = [''.join(numbers[-1].split(',')), ''.join(numbers[-2].split(','))] if numbers else None
    else:
        f_matches = [''.join(numbers[-1].split(',')), None] if numbers else None
    if len(matches) == 1: 
        s_matches = re.findall(r"[\d.]+", matches[-1])
        f_matches = s_matches
    elif len(matches) > 1:
        s_matches = re.findall(r"[\d.]+", matches[-1])
    else:
        s_matches = matches
    return s_matches, f_matches

def process_target(target, dataset):
    if dataset == 'gsm8k':
        target = int(target.split('#### ')[-1].replace(',',''))
        return target
    elif dataset == 'aime':
        return target 
    else:
        raise ValueError('Unknown dataset')
    
def evaluation(result_file):
    results = []
    f = open(result_file, 'r')
    for line in f:
        results.append(json.loads(line))
    f.close()

    total, s_correct, f_correct = 0, 0, 0
    total_time, total_token, total_useful_token = 0, 0, 0
    
    total_steps = 0
    total_experts = 0
    for idx, problem in enumerate(results):

        answer = problem['answer'][0]
        strict_predict_ans, flexible_predict_ans = extract_boxed_text(answer)
        target = problem['task']['answerKey']

        # target = process_target(target, 'gsm8k')

        total += 1
        
        if len(strict_predict_ans) > 0 or (flexible_predict_ans is not None and len(flexible_predict_ans) > 0): 
            try:
                correct_flag = False
                if len(strict_predict_ans) > 0 and str(strict_predict_ans[0]) == str(target):
                    s_correct += 1
                    f_correct += 1
                    correct_flag = True
                elif str(flexible_predict_ans[0]) == str(target):
                    f_correct += 1
                    correct_flag = True
                elif args.relax and flexible_predict_ans[1] is not None and str(flexible_predict_ans[1]) == str(target):
                    f_correct += 1
                    correct_flag = True             

                #print(f"Correct = {correct_flag}, \tStrict Predict = {strict_predict_ans},\t Flexible Predict = {flexible_predict_ans}, \tTarget = {target}")
            except:
                #print('Error in extracting answers: ', strict_predict_ans, flexible_predict_ans)
                pass
        else:
            #print('No answer found: ', idx, target)
            pass

        total_time += problem['time']
        total_token += problem['tokens']
        total_useful_token += problem['useful_tokens']

        total_steps += problem['steps']
        total_experts += problem.get('unique_experts_count', 0)
     
    return {
        'strict_accuracy': s_correct / total,
        'soft_accuracy': f_correct / total,
        'strict_match': s_correct,
        'soft_match': f_correct,
        'total': total,
        'total_time': total_time,
        'total_token': total_token,
        'avg_token': total_token / total,
        'avg_useful_token': total_useful_token / total,
        'token/s': total_token / total_time,
        'useful_token/s': total_useful_token / total_time,
        'avg_steps': total_steps / total if total > 0 else -1,
        'unique_experts': total_experts / total if total > 0 else -1,
    }

def passk_evaluation(result_files):
    """
    Pass@k evaluation for multiple-choice tasks using answerKey.
    A question is counted as correct if ANY of the result files answers it correctly.

    result_files: list of paths to JSONL result files.
    """

    # Load all files
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

        target = all_results[0][qid]['task']['answerKey']

        strict_hit = False
        soft_hit = False

        for fidx in range(num_files):

            answer = all_results[fidx][qid]['answer'][0]
            strict_ans, flexible_ans = extract_boxed_text(answer)

            try:
                # strict match
                if strict_ans is not None and len(strict_ans) > 0 and str(strict_ans[0]) == str(target):
                    strict_hit = True
                    soft_hit = True
                    break

                # soft match
                if flexible_ans is not None and len(flexible_ans) > 0 and str(flexible_ans[0]) == str(target):
                    soft_hit = True
                    break

                # relaxed flexible match
                if flexible_ans is not None and len(flexible_ans) > 1 and \
                   args.relax and flexible_ans[1] is not None and str(flexible_ans[1]) == str(target):
                    soft_hit = True
                    break

            except Exception:
                continue

        # count correctness
        if strict_hit:
            strict_correct += 1
            soft_correct += 1
        elif soft_hit:
            soft_correct += 1

    # final output
    return {
        'strict_accuracy': strict_correct / num_questions,
        'soft_accuracy': soft_correct / num_questions,
        'strict_match': strict_correct,
        'soft_match': soft_correct,
        'total': num_questions,
    }


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--result_file', type=str, default='results.csv')
    args.add_argument('--dataset', type=str, default='gsm8k')
    args.add_argument('--split', type=str, default='test')
    args.add_argument('--num-samples', type=int, default=None)
    args = args.parse_args()

    result_file = args.result_file    
    if args.dataset == 'gsm8k':
        dataset = load_dataset('openai/gsm8k', 'main')
        dataset_type = args.split
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
        
        if len(strict_predict_ans) > 0 or (flexible_predict_ans is not None and len(flexible_predict_ans) > 0): 
            try:
                correct_flag = False
                if len(strict_predict_ans) > 0 and float(strict_predict_ans[0]) == float(target):
                    s_correct += 1
                    f_correct += 1
                    correct_flag = True
                elif float(flexible_predict_ans[0]) == float(target):
                    f_correct += 1
                    correct_flag = True
                elif args.relax and flexible_predict_ans[1] is not None and float(flexible_predict_ans[1]) == float(target):
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

