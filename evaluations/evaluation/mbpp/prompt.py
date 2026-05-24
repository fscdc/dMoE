import os
from collections import defaultdict
from typing import Any

import io
import numpy as np
from tqdm import tqdm

instruction = """
You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:\n\n{test_list_0}\n{test_list_1}\n{test_list_2}\nYour response should only contain the code for this function.\n
"""

def format_prompt(row: dict):
    prompt_messages = instruction.format(
        text=row['text'],  
        test_list_0=row['test_list'][0],
        test_list_1=row['test_list'][1],
        test_list_2=row['test_list'][2],
    )
    return prompt_messages, row['code']