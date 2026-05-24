"""
https://github.com/openai/simple-evals/blob/main/common.py
"""

import os
from collections import defaultdict
from typing import Any

import io
import random
import numpy as np
from tqdm import tqdm

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. Think step by step and then give the final answer of the following format: \\boxed{{$LETTER}} where LETTER is one of ABCD.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

def format_prompt(row: dict):
    choices = [
        row["Correct Answer"],
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
    ]
    permutation = [0, 1, 2, 3]
    random.shuffle(permutation)

    choices = [choices[i] for i in permutation]
    correct_index = choices.index(row["Correct Answer"])
    correct_answer = "ABCD"[correct_index]
    choices_dict = dict(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=row["Question"]
    )
    prompt_messages = QUERY_TEMPLATE_MULTICHOICE.format(**choices_dict)
    return prompt_messages, correct_index