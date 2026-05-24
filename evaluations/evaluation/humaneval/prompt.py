'''
https://github.com/openai/simple-evals/blob/main/humaneval_eval.py
'''

import os
from collections import defaultdict
from typing import Any

import io
import numpy as np
from tqdm import tqdm

instruction = "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n"

def format_prompt(row: dict):
    prompt_messages = instruction + row['prompt']
    return prompt_messages, row['canonical_solution']