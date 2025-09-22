import sys
sys.path.append("/media/16TBNVME/home/amaand/mlf2/gepa/src")

import gepa
from openai import OpenAI
from gepa.examples.testgen.init_dataset import init_dataset
from gepa.adapters.testgen_adapter.testgen_adapter import TestGenAdapter

trainset, valset = init_dataset(train_size=75, val_size=25)
def reflection_lm(prompt: str) -> str:
    client = OpenAI(
        api_key="serving-on-vllm",
        base_url="http://localhost:8190/v1"
    )

    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
    )

    return (resp.choices[0].message.content or "").strip()


adapter = TestGenAdapter(
    model='openai/gpt-oss-20b',
    api_base='http://localhost:7190/v1',
    api_key='serving-on-vllm',
    reflection_lm='openai/gpt-oss-120b'
)

print('-' * 200)

result = gepa.optimize(
    seed_candidate={
        "system_prompt": 
        """
        Given the following coding problem:
        {question}
        Can you generate 5 different input and outputs pair test cases for the above problem in a JSON object.
        Important Note: Make sure the length of the inputs and outputs are EQUAL so have EXACTLY 5 INPUTS AND 5 OUTPUTS.
        Make sure every KEY AND ELEMENT OF THE ARRAY in the JSON is wrapped in DOUBLE QUOTES.
        Generate test cases and wrap your final answer in a JSON code block WITHOUT ANY WHITESPACES OR EXTRA \\n characters.
        Example format:
        ```json{{"inputs":["1","2"],"outputs":["4","5"]}}```""",

        "rubric_prompt":
        """The test cases should check for edge cases that would pass for a valid solution but fail for an almost correct solution.
        The JSON must be a dictionary with two keys: 'inputs' and 'outputs'.
        'inputs' and 'outputs' must both be lists.
        Elements may be numbers, strings, or nested lists etc depending on the problem.
        """
        },
    trainset=trainset,
    valset=valset,
    adapter=adapter,
    reflection_lm=reflection_lm,
    perfect_score=3,
    skip_perfect_score=False,
    use_wandb=False,
    max_metric_calls=125,
    seed=12345,
    display_progress_bar=True,
)

print("Final GEPA Optimized Rubric Prompt:\n", result.best_candidate["rubric_prompt"])