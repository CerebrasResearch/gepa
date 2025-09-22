
from pathlib import Path

import json, random
from tqdm import tqdm
import random
import json
from openai import OpenAI
from tqdm import tqdm
import sys
sys.path.append("/media/16TBNVME/home/amaand/mlf2/gepa/src")

def init_dataset(samples_dir: str = "samples_with_inc_code", train_size: int = 75, val_size: int = 25):
    """
    Load dataset of question, gold_code, incorrect_code
    from a folder of jsonl files.

    Returns:
        trainset, valset, testset
    """
    all_examples = []
    samples = Path(samples_dir)
    count = 0
    total = train_size + val_size
    files = list(samples.glob("*.jsonl"))

    for i, filepath in enumerate(tqdm(files, desc="Processing JSONL files", total=total)):
        if count == total:
            break
        with filepath.open("r", encoding="utf-8") as f:
            obj = json.loads(f.readline())
            incorrect_codes = obj.get("incorrect_codes")

            if len(incorrect_codes) == 0:
                continue

            gt = json.loads(obj.get("ground_truth"))
            fn_name = gt.get('fn_name', None)

            all_examples.append({
                "question": obj.get("question"),
                "gold_code": obj.get("solutions")[0],
                "incorrect_code": incorrect_codes[0],
                "fn_name": fn_name
            })
            count += 1

    trainset = all_examples[:train_size]
    valset = all_examples[train_size:]

    return trainset, valset
