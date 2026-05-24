import json
import re


def extract_boxed_text_sampling(document):
    solution = str(document)

    #===========================
    # 1. Strict match: \boxed{A}
    #===========================
    strict_prediction = None
    prediction_match = re.search(r"\\boxed\{([^}]*)\}", solution)

    if prediction_match:
        content = prediction_match.group(1)
        choice_match = re.search(r'\b([ABCD])\b', content)
        if choice_match:
            strict_prediction = choice_match.group(1)

    #=================================
    # 2. Flexible match: fallback rule
    #=================================
    flexible_prediction = None

    # Only try fallback if strict is None
    if strict_prediction is None:
        patterns = [
            r"(?i)Answer[ \t]*:[ \t]*([A-D])",
            r"(?i)Answer is[ \t]*:?[ \t]*([A-D])",
            r"(?i)is option[ \t]*:?[ \t]*([A-D])",
            r"(?i)\*\*Answer:\*\*[ \t]*([A-D])",
            r"(?i)Option ([A-D])",
        ]
        for pattern in patterns:
            prediction_match = re.search(pattern, solution)
            if prediction_match:
                flexible_prediction = prediction_match.group(1)
                break

    # If strict prediction exists, flexible = strict
    if strict_prediction is not None:
        flexible_prediction = strict_prediction

    return strict_prediction, flexible_prediction


def extract_boxed_text(document, expected_answer):
    solution = document
    correct = False

    # Extract prediction wrapped by "\\boxed{}"
    prediction_match = re.search(r"\\boxed\{([^}]*)\}", str(solution))

    #re.search(r'\\boxed\{([^}]*)\}', text)
    prediction = None
    if prediction_match:
        content = prediction_match.group(1)
        # Find the first occurrence of A, B, C, or D inside the boxed content
        choice_match = re.search(r'\b([ABCD])\b', content)
        if choice_match:
            prediction = choice_match.group(1)
        
        #prediction = prediction_match[-1]
        # print(solution[0][0][-100:])
    if prediction is None:
        patterns = [ 
            r"(?i)Answer[ \t]*:[ \t]*([A-D])",
            r"(?i)Answer is[ \t]*:?[ \t]*([A-D])",
            r"(?i)is option[ \t]*:?[ \t]*([A-D])",
            r"(?i)\*\*Answer:\*\*[ \t]*([A-D])",
            r"(?i)Option ([A-D])",
        ]
        for pattern in patterns:
            prediction_match = re.search(pattern, str(solution))
            if prediction_match:
                prediction = prediction_match.group(1)
                break
        
    
    # Check if prediction matches the expected answer
    if prediction is not None: #prediction == expected_answer:
        try:
            if prediction.lower() == "ABCD"[expected_answer].lower():
                correct = True
        except ValueError:
            pass
    
    #print(f"Correct= {correct},\t Prediction = {prediction},\t Expected = {expected_answer},\t Match = {prediction_match}")
    return correct

def evaluation(result_file):
    results = []
    f = open(result_file, 'r')
    for line in f:
        results.append(json.loads(line))
    f.close()

    total, s_correct = 0, 0
    total_time, total_token = 0, 0

    total_steps = 0
    total_experts = 0
    
    for idx, problem in enumerate(results):

        expected_answer = problem['answer_index']
        correctness = extract_boxed_text(problem['answer'], expected_answer)

        total += 1
        total_time += problem['time']
        total_token += problem['tokens']
        s_correct += correctness

        total_steps += problem.get('steps', 0)
        total_experts += problem.get('unique_experts_count', 0)

    print(f"Strict Match Accuracy = {s_correct}/{total}  = {s_correct/total}")
    return {
        'strict_accuracy': s_correct / total,
        'strict_match': s_correct,
        'total': total,
        'total_time': total_time,
        'total_token': total_token,
        'token/s': total_token / total_time,
        'avg_steps': total_steps / total if total > 0 else 0,
        'unique_experts': total_experts / total if total > 0 else 0,
    }

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