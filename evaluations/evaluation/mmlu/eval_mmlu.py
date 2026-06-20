import json
import re


def extract_answer(solution):
    """
    Extract predicted answer letter (A/B/C/D) from model output.
    Returns (prediction, confidence) where confidence is one of:
      'last_line'   - last line matches prompt format "Answer: X"
      'answer_tag'  - "Answer: X" found anywhere (last occurrence)
      'boxed'       - \boxed{X}
      'explicit'    - explicit phrases like "the answer is X", "correct answer is X"
      'pattern'     - weaker patterns like "Option X", "(X)", "**X**"
      'last_letter' - last standalone letter A-D in text (lowest confidence)
      None          - no answer found
    """
    solution = str(solution)

    # 1. Last line: "Answer: X" — matches prompt format exactly
    last_line = solution.strip().split('\n')[-1]
    m = re.search(r'(?i)Answer\s*:\s*\**\s*([A-D])\b', last_line)
    if m:
        return m.group(1).upper(), 'last_line'

    # 2. "Answer: X" anywhere — take last occurrence
    matches = re.findall(r'(?i)Answer\s*:\s*\**\s*([A-D])\b', solution)
    if matches:
        return matches[-1].upper(), 'answer_tag'

    # 3. \boxed{A}
    m = re.search(r'\\boxed\{([^}]*)\}', solution)
    if m:
        cm = re.search(r'\b([ABCD])\b', m.group(1))
        if cm:
            return cm.group(1).upper(), 'boxed'

    # 4. Explicit answer phrases — take last occurrence
    explicit_patterns = [
        r'(?i)(?:the\s+)?correct\s+answer\s+is\s*:?\s*\**([A-D])\b',
        r'(?i)(?:the\s+)?answer\s+is\s*:?\s*\**([A-D])\b',
        r'(?i)(?:so|thus|therefore)[,\s]+(?:the\s+)?(?:correct\s+)?answer\s+is\s*:?\s*\**([A-D])\b',
        r'(?i)I\s+(?:would\s+)?(?:choose|select|pick)\s+:?\s*\**([A-D])\b',
    ]
    for pattern in explicit_patterns:
        matches = re.findall(pattern, solution)
        if matches:
            return matches[-1].upper(), 'explicit'

    # 5. Weaker structural patterns — take last occurrence
    weak_patterns = [
        r'(?i)is\s+option\s*:?\s*([A-D])\b',
        r'(?i)\*\*Answer:\*\*\s*([A-D])\b',
        r'(?i)Option\s+([A-D])\b',
        r'\(([A-D])\)',                          # (A), (B), ...
        r'(?i)\b([A-D])\s+is\s+(?:correct|right)\b',
    ]
    for pattern in weak_patterns:
        matches = re.findall(pattern, solution)
        if matches:
            return matches[-1].upper(), 'pattern'

    # 6. Last standalone letter A-D in entire text (lowest confidence fallback)
    matches = re.findall(r'\b([A-D])\b', solution)
    if matches:
        return matches[-1].upper(), 'last_letter'

    return None, None


# Confidence tiers for strict vs flexible accuracy
_STRICT_CONFIDENCE = {'last_line', 'answer_tag', 'boxed', 'explicit'}
_FLEXIBLE_CONFIDENCE = {'last_line', 'answer_tag', 'boxed', 'explicit', 'pattern', 'last_letter'}


def extract_boxed_text_sampling(document):
    """Returns (strict_prediction, flexible_prediction) for sampling / pass@k use."""
    prediction, confidence = extract_answer(document)
    if confidence in _STRICT_CONFIDENCE:
        return prediction, prediction
    elif confidence in _FLEXIBLE_CONFIDENCE:
        return None, prediction
    return None, None


def extract_boxed_text(document, expected_answer):
    """Returns bool: whether the extracted answer matches expected_answer (0-indexed int)."""
    prediction, _ = extract_answer(document)
    if prediction is None:
        return False
    try:
        return prediction.upper() == "ABCD"[expected_answer].upper()
    except (IndexError, TypeError):
        return False


def evaluation(result_file):
    results = []
    with open(result_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))

    total = 0
    s_correct = 0   # strict: last_line / answer_tag / boxed / explicit
    f_correct = 0   # flexible: all confidence levels
    no_answer = 0

    total_time, total_token = 0, 0
    total_steps = 0
    total_experts = 0

    confidence_counts = {}

    for problem in results:
        expected_answer = problem['answer_index']
        answer_text = problem['answer']

        prediction, confidence = extract_answer(answer_text)

        total += 1
        total_time += problem['time']
        total_token += problem['tokens']
        total_steps += problem.get('steps', 0)
        total_experts += problem.get('unique_experts_count', 0)

        confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1

        if prediction is None:
            no_answer += 1
            continue

        correct = prediction.upper() == "ABCD"[expected_answer].upper()

        if correct:
            if confidence in _STRICT_CONFIDENCE:
                s_correct += 1
                f_correct += 1
            else:
                f_correct += 1

    print(f"Strict  Accuracy = {s_correct}/{total} = {s_correct/total:.4f}  (last_line/answer_tag/boxed/explicit)")
    print(f"Flexible Accuracy = {f_correct}/{total} = {f_correct/total:.4f}  (all patterns)")
    print(f"No answer found   = {no_answer}/{total}")
    print(f"Confidence breakdown: {confidence_counts}")

    return {
        'strict_accuracy': s_correct / total,
        'flexible_accuracy': f_correct / total,
        'strict_match': s_correct,
        'flexible_match': f_correct,
        'total': total,
        'no_answer': no_answer,
        'total_time': total_time,
        'total_token': total_token,
        'token/s': total_token / total_time,
        'avg_steps': total_steps / total if total > 0 else 0,
        'unique_experts': total_experts / total if total > 0 else 0,
        'confidence_breakdown': confidence_counts,
    }


def time_evaluation(result_file):
    results = []
    with open(result_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))

    total_time, total_token, total = 0, 0, 0
    for problem in results:
        total += 1
        total_time += problem['time']
        total_token += problem['tokens']

    return {
        'total': total,
        'total_time': total_time,
        'total_token': total_token,
        'token/s': total_token / total_time,
    }
