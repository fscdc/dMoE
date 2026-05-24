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
    # Regular expression to find text within \boxed{}
    pattern = r"\\boxed\{([^}]*)\}"
    #pattern = r'#### \s*(.*)'
    matches = re.findall(pattern, document)
    #if len(matches) != 1: 
        #print(document)
        #print("Flexible Matching: ")
    numbers = re.findall(r'\b\d+\.?\d*\b', document)
    f_matches = [numbers[-1]] if numbers else None
    if len(matches) == 1: 
        s_matches = re.findall(r"[\d.]+", matches[-1])
        f_matches = s_matches
    elif len(matches) > 1:
        s_matches = re.findall(r"[\d.]+", matches[-1])
    else:
        s_matches = matches
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

        target = problem['task']['answer']
        target = process_target(target, 'gsm8k')

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
    result_files: list of file paths, each containing model outputs in GSM8K JSONL format.
    Pass@k logic: a question is counted as correct if ANY of the result files produces the correct answer.
    """

    # Load all result files
    all_results = []
    for file in result_files:
        with open(file, "r") as f:
            results = [json.loads(line) for line in f]
            all_results.append(results)

    # Number of questions assumed identical across files
    num_files = len(all_results)
    num_questions = len(all_results[0])

    strict_correct = 0
    soft_correct = 0

    for qid in range(num_questions):
        target = all_results[0][qid]['task']['answer']
        target = process_target(target, "gsm8k")

        strict_hit = False
        soft_hit = False

        # Check all files for this question
        for fidx in range(num_files):
            answer = all_results[fidx][qid]['answer'][0]
            strict_ans, flexible_ans = extract_boxed_text(answer)

            try:
                # Strict match
                if len(strict_ans) > 0 and float(strict_ans[0]) == float(target):
                    strict_hit = True
                    soft_hit = True
                    break

                # Flexible match
                if flexible_ans is not None and len(flexible_ans) > 0 and float(flexible_ans[0]) == float(target):
                    soft_hit = True
                    break

                # Relaxed match if allowed
                if args.relax and flexible_ans is not None and flexible_ans[1] is not None and float(flexible_ans[1]) == float(target):
                    soft_hit = True
                    break

            except:
                pass

        # Count correctness
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

def plot_entropy_curves_multiple_steps(
    sampling_data,
    cases,
    num_sampling: int,
    start_step: int = 1,
    step_interval: int = 4,
    save_dir: str = "./figs",
):
    """
    Plot and save average token-wise entropy curves for multiple decoding steps.

    For each step:
      - x-axis: token index
      - y-axis: mean entropy
      - one curve per case

    Additionally:
      - plot mean entropy of committed tokens at this step

    Style:
      - same color for the same label
      - solid line: all tokens
      - dashed line: committed tokens
      - legend: color = label, linestyle = meaning
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    os.makedirs(save_dir, exist_ok=True)

    for step_idx in range(start_step, 100, step_interval):
        print(f"[Plot] Decoding step {step_idx}")
        plt.figure(figsize=(8, 5))

        color_map = {}
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color_idx = 0

        for label, items in cases.items():
            all_entropies = []
            commit_entropy_by_pos = {}

            for ex in items:
                idx = ex["idx"]

                for k in range(num_sampling):
                    sample = sampling_data[k][idx]
                    records = sample.get("records", None)
                    if records is None or step_idx >= len(records):
                        continue

                    record = records[step_idx]
                    entropy = record.get("entropy", None)
                    commit_pos = record.get("commit_pos", [])

                    if entropy is None:
                        continue

                    entropy_vec = entropy[0]
                    all_entropies.append(entropy_vec)

                    for pos in commit_pos:
                        if pos < len(entropy_vec):
                            commit_entropy_by_pos.setdefault(pos, []).append(
                                entropy_vec[pos]
                            )

            if len(all_entropies) == 0:
                continue

            all_entropies = np.array(all_entropies)
            mean_entropy = all_entropies.mean(axis=0)

            # truncate x-axis
            T = len(mean_entropy)
            if T == 128:
                max_len = 80
            elif T == 256:
                max_len = 160
            else:
                max_len = T

            # assign consistent color per label
            if label not in color_map:
                color_map[label] = color_cycle[color_idx % len(color_cycle)]
                color_idx += 1
            color = color_map[label]

            # all tokens — solid line
            plt.plot(
                range(max_len),
                mean_entropy[:max_len],
                color=color,
                linewidth=2,
            )

            # commit tokens — dashed line
            if len(commit_entropy_by_pos) > 0:
                commit_positions = [
                    p for p in sorted(commit_entropy_by_pos.keys()) if p < max_len
                ]
                commit_mean_entropy = [
                    np.mean(commit_entropy_by_pos[p]) for p in commit_positions
                ]

                plt.plot(
                    commit_positions,
                    commit_mean_entropy,
                    color=color,
                    linestyle="--",
                    marker="o",
                    markersize=4,
                )

        # axis labels
        plt.xlabel("Token Position")
        plt.ylabel("Entropy")

        # -------- legends --------
        # legend 1: label (color)
        label_handles = [
            Line2D([0], [0], color=color, linewidth=2)
            for label, color in color_map.items()
        ]
        label_names = [label.replace("_", " ") for label in color_map.keys()]

        legend1 = plt.legend(
            label_handles,
            label_names,
            loc="upper right",
            title="Case",
        )

        # legend 2: line style (meaning)
        style_handles = [
            Line2D([0], [0], color="black", linewidth=2, linestyle="-"),
            Line2D([0], [0], color="black", linewidth=2, linestyle="--"),
        ]
        style_names = ["All tokens", "Committed tokens"]

        legend2 = plt.legend(
            style_handles,
            style_names,
            loc="upper center",
            frameon=False,
        )

        # plt.gca().add_artist(legend1)

        plt.tight_layout()

        save_path = f"{save_dir}/entropy_curves_step_{step_idx}_sampling_{num_sampling}.pdf"
        plt.savefig(save_path)
        plt.close()

        print(f"  Saved to {save_path}")


def evaluation_with_voting(
    sampling_dir: str,
    prefix: str,
    suffix: str,
    baseline_file: str,
):
    import json
    import os
    from glob import glob
    from collections import Counter
    import matplotlib.pyplot as plt

    HIGH_ENTROPY_THRESH = 0.9

    LABELS = [
        "BASELINE_WRONG_VOTING_CORRECT",
        "BASELINE_CORRECT_VOTING_WRONG",
        "BOTH_CORRECT",
        "BOTH_WRONG",
    ]

    # ===== load baseline =====
    baseline_data = []
    with open(baseline_file, "r") as f:
        for line in f:
            baseline_data.append(json.loads(line))

    num_samples = len(baseline_data)

    # ===== load sampling files =====
    sampling_files = sorted(glob(os.path.join(sampling_dir, f"{prefix}*{suffix}.json")))
    assert len(sampling_files) > 0

    sampling_data = []
    for file in sampling_files:
        rows = []
        with open(file, "r") as f:
            for line in f:
                rows.append(json.loads(line))
        assert len(rows) == num_samples
        sampling_data.append(rows)

    num_sampling = len(sampling_data)
    print(f"Loaded {num_sampling} sampling files + 1 baseline file")

    # ===== counters =====
    cnt_baseline_wrong_voting_correct = 0
    cnt_baseline_correct_voting_wrong = 0
    cnt_both_correct = 0
    cnt_both_wrong = 0
    total = 0

    cases = {lab: [] for lab in LABELS}
    vote_strength_stats = {lab: Counter() for lab in LABELS}

    # ===== high-entropy commit stats (existing) =====
    high_entropy_commit_stats = [
        {
            "high_by_label": {lab: 0 for lab in LABELS},
            "total_by_label": {lab: 0 for lab in LABELS},
        }
        for _ in range(num_sampling)
    ]

    # ===== avg step (existing) =====
    step_sum = {lab: 0 for lab in LABELS}
    step_cnt = {lab: 0 for lab in LABELS}

    # ===== step distribution (existing) =====
    step_values = {lab: [] for lab in LABELS}

    # ===== NEW: high-entropy token count (for mean) =====
    high_entropy_token_sum = {lab: 0 for lab in LABELS}
    high_entropy_token_cnt = {lab: 0 for lab in LABELS}

    # ===== main loop =====
    for idx in range(num_samples):
        problem = baseline_data[idx]
        target = process_target(problem["task"]["answer"], "gsm8k")

        base_ans = problem["answer"][0]
        s_base, f_base = extract_boxed_text(base_ans)

        baseline_correct = False
        try:
            if s_base:
                baseline_correct = float(s_base[0]) == float(target)
            elif f_base:
                baseline_correct = float(f_base[0]) == float(target)
        except:
            baseline_correct = False

        votes = []
        for k in range(num_sampling):
            sample = sampling_data[k][idx]
            for ans in sample["answer"]:
                s, f = extract_boxed_text(ans)
                if s:
                    votes.append(s[0])
                    break
                elif f:
                    votes.append(f[0])
                    break

        if votes:
            voted_pred, most_common_count = Counter(votes).most_common(1)[0]
            try:
                voting_correct = float(voted_pred) == float(target)
            except:
                voting_correct = False
        else:
            voting_correct = False
            most_common_count = 0

        if (not baseline_correct) and voting_correct:
            label = "BASELINE_WRONG_VOTING_CORRECT"
            cnt_baseline_wrong_voting_correct += 1

            # ===== NEW: print question =====
            question = problem["task"].get("question", "")
            print(f"\n[BASELINE_WRONG_VOTING_CORRECT] idx={idx}")
            print(question)

        elif baseline_correct and (not voting_correct):
            label = "BASELINE_CORRECT_VOTING_WRONG"
            cnt_baseline_correct_voting_wrong += 1
        elif baseline_correct and voting_correct:
            label = "BOTH_CORRECT"
            cnt_both_correct += 1
        else:
            label = "BOTH_WRONG"
            cnt_both_wrong += 1

        cases[label].append({"idx": idx})
        vote_strength_stats[label][most_common_count] += 1
        total += 1

        # ===== per-sampling analysis =====
        for k in range(num_sampling):
            sample = sampling_data[k][idx]
            records = sample.get("records", None)

            high_entropy_commit_stats[k]["total_by_label"][label] += 1
            if records is None:
                continue

            high_entropy_token_count = 0
            has_high_entropy_commit = False

            for record in records:
                entropy = record.get("entropy", None)
                commit_pos = record.get("commit_pos", [])
                if entropy is None or not commit_pos:
                    continue

                entropy_vec = entropy[0]
                for pos in commit_pos:
                    if pos < len(entropy_vec) and entropy_vec[pos] >= HIGH_ENTROPY_THRESH:
                        high_entropy_token_count += 1
                        has_high_entropy_commit = True

            if has_high_entropy_commit:
                high_entropy_commit_stats[k]["high_by_label"][label] += 1

            # avg step
            step_sum[label] += len(records)
            step_cnt[label] += 1
            step_values[label].append(len(records))

            # ===== NEW: high-entropy token stats =====
            high_entropy_token_sum[label] += high_entropy_token_count
            high_entropy_token_cnt[label] += 1

    # ===== reports =====
    print("\n========== Voting Strength Distribution ==========")
    for lab in LABELS:
        print(f"\n[{lab}]")
        for k in sorted(vote_strength_stats[lab], reverse=True):
            print(f"  {k}/{num_sampling}: {vote_strength_stats[lab][k]}")

    print(f"\n========== High-Entropy Commit Statistics (threshold = {HIGH_ENTROPY_THRESH}) ==========")
    for k in range(num_sampling):
        print(f"\n[Sampling File {k}]")
        for lab in LABELS:
            high_cnt = high_entropy_commit_stats[k]["high_by_label"][lab]
            lab_total = high_entropy_commit_stats[k]["total_by_label"][lab]
            ratio = high_cnt / lab_total if lab_total > 0 else 0.0
            print(f"  {lab}: high={high_cnt} / total={lab_total} ({ratio:.2%})")

    print("\n========== Average Decoding Steps per Label ==========")
    for lab in LABELS:
        if step_cnt[lab] > 0:
            print(f"  {lab}: avg_steps = {step_sum[lab] / step_cnt[lab]:.2f}")
        else:
            print(f"  {lab}: avg_steps = N/A")

    print("\n========== Average High-Entropy Token Count per Label ==========")
    for lab in LABELS:
        if high_entropy_token_cnt[lab] > 0:
            avg_cnt = high_entropy_token_sum[lab] / high_entropy_token_cnt[lab]
            print(f"  {lab}: avg_high_entropy_tokens = {avg_cnt:.2f}")
        else:
            print(f"  {lab}: avg_high_entropy_tokens = N/A")

    # ===== existing plots =====
    # 过滤掉其中两个case
    cases = {lab: cases[lab] for lab in LABELS if lab != "BOTH_WRONG" and lab != "BASELINE_CORRECT_VOTING_WRONG"}

    plot_entropy_curves_multiple_steps(
        sampling_data=sampling_data,
        cases=cases,
        num_sampling=num_sampling,
        start_step=0,
        step_interval=2,
        save_dir="./figs",
    )

    return {
        "total": total,
        "baseline_wrong_voting_correct": cnt_baseline_wrong_voting_correct,
        "baseline_correct_voting_wrong": cnt_baseline_correct_voting_wrong,
        "both_correct": cnt_both_correct,
        "both_wrong": cnt_both_wrong,
        "vote_strength_stats": vote_strength_stats,
        "high_entropy_commit_stats": high_entropy_commit_stats,
        "avg_decoding_steps": {
            lab: (step_sum[lab] / step_cnt[lab] if step_cnt[lab] > 0 else None)
            for lab in LABELS
        },
        "step_distribution": step_values,
        "avg_high_entropy_token_count": {
            lab: (
                high_entropy_token_sum[lab] / high_entropy_token_cnt[lab]
                if high_entropy_token_cnt[lab] > 0
                else None
            )
            for lab in LABELS
        },
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

