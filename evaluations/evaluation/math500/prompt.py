import os
from collections import defaultdict
from typing import Any

import io
import numpy as np
from tqdm import tqdm


QUERY_TEMPLATE_MATH = """
Please answer the question step by step and put the answer in \\boxed{{}}. {question}
""".strip()

def format_prompt(row: dict):
    prompt_messages = QUERY_TEMPLATE_MATH.format(
        question=row['problem']
    )
    return prompt_messages, row['answer']
