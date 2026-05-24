import os
from collections import defaultdict
from typing import Any

import io
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
    choices = row["choices"]["text"]

    choices = choices + ["NA"] * (4 - len(choices))

    prompt_messages = QUERY_TEMPLATE_MULTICHOICE.format(
        Question=row["question"],
        A=choices[0],
        B=choices[1],
        C=choices[2],
        D=choices[3],
    )
    return prompt_messages, row["answerKey"]