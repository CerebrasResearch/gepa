import random
from datasets import load_dataset
import argparse
import json
from openai import OpenAI
from tqdm import tqdm
import sys
sys.path.append("/media/16TBNVME/home/michaelw/mlf2/inference-time-compute/gepa/src")

from gepa import optimize
from gepa.adapters.cepo_adapter.cepo_coding_adapter import CepoCodingAdapter
from gepa.adapters.cepo_adapter.cepo_utils import llm_call_reason_effort_fallback
from gepa.adapters.cepo_adapter.coding_utils import taco_data_converter


def init_dataset():
    train_split = []
    coding_dataset = load_dataset("Skywork/Skywork-OR1-RL-Data", split="code")
    training_sources = [
        # "train-code-leetcode-Hard", # 527 examples
        "train-code-taco-hard", # 1278 examples
        "train-code-taco-very_hard", # 563 examples
    ]

    training_dataset = coding_dataset.filter(lambda example: example["data_source"] in training_sources)

    for item in training_dataset:
        try:
            converted_item = taco_data_converter(item)
        except:
            continue
        train_split.append(converted_item)

    random.Random(0).shuffle(train_split)


    trainset = train_split[: len(train_split) // 2]
    valset = train_split[len(train_split) // 2 :]

    return trainset, valset


if __name__ == "__main__":
    candidate = {
        "cepo_planning_prompt": "To answer this question, can you come up with a concise plan to solve it step-by-step but do not provide the final answer. Also, for each step, provide your confidence in the correctness of that step as well as your ability to execute it correctly. ",
        "cepo_execution_prompt": "Can you execute the above plan step-by-step to produce the final answer. Be extra careful when executing steps where your confidence is lower.",
        "cepo_reflection_prompt": "Can you review your last N responses and identify any inconsistency between them. After that, can you address it and present a final step-by-step solution to the problem?",
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--budget", type=int, default=400, help="The budget for the optimization process."
    )
    parser.add_argument(
        "--reflection_model_name", type=str, default="openai/gpt-oss-20b", help="The name of the reflection LM to use."
    )
    parser.add_argument(
        "--adapter_model_name", type=str, default="openai/gpt-oss-20b", help="The name of the adapter LM to use."
    )
    parser.add_argument(
        "--adapter_api_base", type=str, default="http://localhost:7190/v1"
    )
    parser.add_argument(
        "--reflection_api_base", type=str, default="http://localhost:7190/v1"
    )
    parser.add_argument(
        "--reflection_minibatch_size", type=int, default=10, help="The size of the minibatch for the reflection LM."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="The seed for the random number generator for reproducibility."
    )
    args = parser.parse_args()
    trainset, valset = init_dataset()
    trainset = random.sample(trainset, k=200) # Limit to 50 samples for demo purposes
    valset   = random.sample(valset, k=50)  # Limit to 50 samples for demo purposes

    print(f"Train set size: {len(trainset)}")
    print(f"Validation set size: {len(valset)}")

    def reflection_lm(prompt: str):
        """Call the reflection language model with the given prompt and return its content string."""
        client = OpenAI(
            api_key="serving-on-vllm",
            base_url=args.reflection_api_base,
            timeout=None,
            max_retries=0,
        )

        response, finish_reason, _ = llm_call_reason_effort_fallback(
                messages=[{"role": "user", "content": prompt}],
                client=client,
                model=args.reflection_model_name,
                max_tokens=None,
                temperature=1.0,
                top_p=1.0,
                reasoning_effort_levels=["high", "medium", "low"]
        )
        return response


    optimized_results = optimize(
        seed_candidate={"cepo_planning_prompt": candidate["cepo_planning_prompt"]},
        trainset=trainset,
        valset=valset,
        adapter=CepoCodingAdapter(
            model=args.adapter_model_name, 
            api_base=args.adapter_api_base, 
            failure_score=0.0,
        ),
        reflection_lm=reflection_lm,
        reflection_minibatch_size=args.reflection_minibatch_size,
        perfect_score=1,
        skip_perfect_score=False,
        use_wandb=False,
        max_metric_calls=args.budget,
        seed=args.seed,
        display_progress_bar=True,
    )

    print(optimized_results.to_dict())