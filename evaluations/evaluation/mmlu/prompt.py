import os
from collections import defaultdict
from typing import Any

import io
import numpy as np
from tqdm import tqdm

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

def format_prompt(row: dict):
    prompt_messages = QUERY_TEMPLATE_MULTICHOICE.format(
        Question=row["question"],
        A=row["choices"][0],
        B=row["choices"][1],
        C=row["choices"][2],
        D=row["choices"][3],
    )
    return prompt_messages, row["answer"]