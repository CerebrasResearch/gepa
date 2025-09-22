import json
from tqdm import tqdm
from pathlib import Path
import re

data_dir = Path("/mlf4-shared/amaand/skywork_code_with_gold_sol_verified/")
output_dir = Path("/data/home/amaand/mlf2/gepa/src/gepa/examples/testgen/samples_with_inc_code")

output_dir.mkdir(parents=True, exist_ok=True)

def extract_python_block(text: str):
    """
    Extract the first Python code block from the text, ignoring <think> reasoning.
    Returns the code as a string.
    """
    # Remove reasoning <think>...</think>
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Look for Python code inside ```python ... ``` fences
    match = re.search(r"```python(.*?)```", cleaned, flags=re.DOTALL)
    if match:
        code = match.group(1).strip()
    else:
        # fallback: try to find indented Python code (lines starting with 4 spaces or a tab)
        match = re.search(r"(?:\n(?: {4}|\t).+)+", cleaned)
        if not match:
            raise ValueError("No Python code found in output")
        code = match.group(0).strip()

    return code

def get_incorrect_codes(plans_and_execs):
    incorrect_codes = []

    for plan in plans_and_execs:
        if plan['plan_success_rate'] < 1:
            execution = None
            for i, score in enumerate(plan['verification']):
                if score == False:
                    execution = plan['executions'][i]
                    break
            if execution:
                incorrect_codes.append(extract_python_block(execution))
            
    return incorrect_codes

def gen_dataset(num_samples = 100):
    count = 0
    for i, filepath in enumerate(tqdm(data_dir.glob("*.jsonl"), desc="Processing JSONL files")):
        output_filepath = output_dir / filepath.name

        if output_filepath.exists():
            count += 1
            continue

        if count == num_samples:
            break

        with filepath.open("r", encoding="utf-8") as f:
            obj = json.loads(f.readline())
            plans_and_execs = obj.get('plans_and_executions')
            incorrect_codes = get_incorrect_codes(plans_and_execs)

            if len(incorrect_codes) != 0:
                obj['incorrect_codes'] = incorrect_codes
                del obj['plans_and_executions']
                del obj['index']
                del obj['source']

                with output_filepath.open("w", encoding="utf-8") as f:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    count += 1

if __name__ == "__main__":
    gen_dataset(100)
