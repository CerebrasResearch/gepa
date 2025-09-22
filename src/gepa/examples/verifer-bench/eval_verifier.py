from train_gepa import init_dataset
import re
from gepa.adapters.anymaths_adapter.anymaths_adapter import AnyMathsStructuredOutput
import json
from gepa.adapters.verifier_adapter.prompts import USER_PROMPT_DEEPSEEK_GRM_2_RESPONSES, NEW_OPTIMIZED_PROMPT, OPTIMIZED_PROMPT_V2

def extract_rating( solution_str):
    # match = re.search(r"Rating:\s*\[\[(\d+)\]\]", solution_str)
    matches = re.findall(r'\\boxed\{(.*?)\}', solution_str[-200:], re.DOTALL)
    if matches:
        try:
            boxed_content = matches[-1]  # take the final \boxed{...}
            rating = list(map(int, re.findall(r'\d+', boxed_content)))
        except Exception:
            rating = [-1, -1]
    else:
        rating = [-1, -1]

    if len(rating) != 2:
        rating = [-1, -1]

    return rating



def post_process_output(output: str, answer: list) -> int:
    # extracts the ratings and the calculates the final score based on the ratings
    extracted_rating = extract_rating(output)
    gt_rating = answer # remapped gt
    
    accuracy = 1
    for pred_val, gt_val in zip(extracted_rating, gt_rating):
        if pred_val == -1:
            accuracy = 0
        pred_label = int(pred_val > 5)
        accuracy &= (pred_label == gt_val)

    return accuracy

if __name__ == "__main__":
    import argparse
    import ast
    from pathlib import Path

    import litellm
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--anymaths_dset_name", type=str, default="dataset_code_100")
    parser.add_argument("--model", type=str, default="ollama/qwen3:4b", help="The model to evaluate.")
    parser.add_argument("--use_api_url", action="store_true", help="Whether to use the API URL.")
    parser.add_argument("--api_url", type=str, default="http://localhost:11434", help="The API URL to use.")
    parser.add_argument("--batch_size", type=int, default=8, help="The batch size for evaluation.")
    parser.add_argument(
        "--max_litellm_workers", type=int, default=1, help="The maximum number of LiteLLM workers to use."
    )
    parser.add_argument(
        "--which_prompt",
        type=str,
        default="seed",
        choices=["seed", "optimized"],
        help="The prompt to use for evaluation.",
    )

    args = parser.parse_args()

    dataset = args.anymaths_dset_name

    api_url = args.api_url

    model = args.model
    max_litellm_workers = args.max_litellm_workers

    _, valset, testset = init_dataset(dataset)

    # testset = valset
    if args.which_prompt == "seed":
        INSTRUCTION_PROMPT_PATH = Path(__file__).parent / "prompt-templates/instruction_prompt.txt"
        USER_PROMPT = USER_PROMPT_DEEPSEEK_GRM_2_RESPONSES
    else:
        INSTRUCTION_PROMPT_PATH = Path(__file__).parent / "prompt-templates/optimal_prompt.txt"
        USER_PROMPT = NEW_OPTIMIZED_PROMPT
        USER_PROMPT = OPTIMIZED_PROMPT_V2

    print('using user prompt:')
    print(USER_PROMPT)
    print('-' * 100)

    instruction = INSTRUCTION_PROMPT_PATH.read_text()

    batched_testset = []
    batch_size = args.batch_size

    for i in range(0, len(testset), batch_size):
        batched_testset.append(testset[i : i + batch_size])

    total_score = 0.0

    print("-" * 100)
    print(f"Evaluating model: {model}")
    print(f"Using API URL: {api_url if api_url else 'No API URL'}")
    print(f"Batch size: {batch_size}")
    print(f"Max LiteLLM workers: {max_litellm_workers}")
    print(f"Using prompt: {args.which_prompt}")
    print("-" * 100)

    all_responses = []   # accumulate everything

    with tqdm(total=len(testset), desc="Evaluating") as pbar:
        for batch in batched_testset:
            litellm_requests = []
            system_content = instruction
            for item in batch:
                question = f"{item['input']}"
                other_data = item["additional_context"]
                messages = [{
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": USER_PROMPT.format(
                        question=question,
                        response1=other_data['response1'],
                        response2=other_data['response2']
                    )
                }]

                litellm_requests.append(messages)

            responses = litellm.batch_completion(
                model=model,
                messages=litellm_requests,
                api_base=api_url,              # <- use the local var you set above
                max_workers=max_litellm_workers,
                api_key="EMPTY",
                drop_params=True,
                stream=False,
            )
            

            for resp_obj, item in zip(responses, batch, strict=False):
                try:
                    resp_text = resp_obj.choices[0].message['content'].strip()
                    all_responses.append({
                        "input": item["input"],
                        "answer": item["additional_context"]["answer"],
                        "response": resp_text
                    })
                    score = post_process_output(resp_text, item['additional_context']['answer'])
                    total_score += score
                except Exception:
                    continue

            pbar.update(len(batch))
            pbar.set_postfix({"Score": f"{total_score} / {len(testset):.4f}"})


    # JSON dump results (write ALL, not just last batch)
    with open("verifier_eval_responses.jsonl", "w", encoding="utf-8") as f:
        for rec in all_responses:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")