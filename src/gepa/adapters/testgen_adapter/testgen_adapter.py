import sys
sys.path.append("/media/16TBNVME/home/amaand/mlf2/gepa/src")

from typing import Any, TypedDict, Optional
import litellm
from pydantic import BaseModel, Field
import re
from gepa.core.adapter import EvaluationBatch, GEPAAdapter
import json
import traceback
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from gepa.adapters.cepo_adapter.coding_utils import calculate_score
from tqdm import tqdm


class TestGenDataInst(TypedDict):
    question: str
    gold_code: str
    incorrect_code: str
    fn_name: Optional[str]


class TestGenTrajectory(TypedDict):
    data: TestGenDataInst
    full_assistant_response: str


class TestGenRolloutOutput(TypedDict):
    full_assistant_response: str


# class TestGenStructuredOutput(BaseModel):
#     inputs: List[Any] = Field(..., description="List of input values for the test cases")
#     outputs: List[Any] = Field(..., description="List of expected outputs for the test cases")


class TestGenAdapter(GEPAAdapter[TestGenDataInst, TestGenTrajectory, TestGenRolloutOutput]):
    """TestGen Adapter is a GEPAAdapter for any dataset that contains mathematical word problems
    of varying complexity and structure. It is designed to handle a wide range of mathematical
    tasks, including arithmetic, algebra, and more.

    Note: Ollama must be installed and configured to use this adapter.
    """

    def __init__(
        self,
        model: str,
        failure_score: float = 0.0,
        api_base: str | None = "http://localhost:7190/v1",
        max_litellm_workers: int = 10,
        api_key: str | None = None,
        reflection_lm: str | None = None
    ) -> None:

        self.model = model
        self.failure_score = failure_score
        self.litellm = litellm
        self.max_litellm_workers = max_litellm_workers
        self.api_base = api_base
        self.api_key = api_key
        self.reflection_lm = reflection_lm
        self.base_client = OpenAI(api_key="serving-on-vllm", base_url="http://localhost:7190/v1")
        self.ref_client = OpenAI(api_key="serving-on-vllm",base_url="http://localhost:8190/v1")
        self.retries = 3

   
    def _extract_json_block(self, text: str):
        text = text.strip()
        
        # Remove code block markers
        if text.startswith('```json'):
            content = text[7:]
        elif text.startswith('```'):
            content = text[3:] 
        else:
            content = text
        
        if content.endswith('```'):
            content = content[:-3]
        
        content = content.strip()
        
        # Simple JSON parsing - should work now
        try:
            return json.loads(content)
        except Exception:
            trace = traceback.format_exc()
            print('Could not extract json block error from content:')
            print(repr(content))
            print('Because of the issue in:')
            print(trace)
            #raise ValueError(f"JSON parse error: {e}")
        
    def _all_true(self, lst):
        return len(lst) > 0 and all(x is True for x in lst)

    def _all_false(self, lst):
        return len(lst) > 0 and all(x is False for x in lst)

    def _only_bools(self, lst):
        return all(isinstance(x, bool) for x in lst) and len(lst) > 0

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """
        Generate improved text for candidate components (only rubric_prompt in our case).
        """
        updated_components = {}

        for component in components_to_update:
            # Only optimize rubric_prompt
            if component != "rubric_prompt":
                continue

            examples = reflective_dataset.get(component, [])
            if not examples:
                continue

            # Format the examples into a readable block
            examples_formatted = ""
            for i, example in enumerate(examples):
                examples_formatted += f"Example {i+1}:\n"
                examples_formatted += f"Question: {example['Inputs']}\n"
                examples_formatted += f"Generated Test Cases: {example['Generated Outputs']}\n"
                examples_formatted += f"Feedback: {example['Feedback']}\n\n"

            # Reflection prompt: ask the LLM to rewrite rubric_prompt
            reflection_prompt = f"""
            I provided the assistant with the following rubric prompt:
            ```
            {candidate['rubric_prompt']}
            ```
            The rubric prompt describes how to generate test cases for coding problems.

            Here are examples of how the prompt was used, with outputs and feedback:
            ```
            {examples_formatted}
            ```

            Your task is to rewrite the rubric prompt so that the assistant generates better test cases.
            The new rubric prompt MUST still instruct the assistant to:
            - Generate 5 test cases in JSON with "inputs" and "outputs" keys.
            - Use double quotes around all elements.
            - Wrap the final answer inside ```json ... ```.

            Only return the improved rubric prompt inside a ``` block.
            """

            try:
                new_content = ""
                for attempt in range(self.retries):
                    resp = self.ref_client.chat.completions.create(
                            model=self.reflection_lm,
                            messages=[{"role": "user", "content": reflection_prompt}],
                            temperature=0.6,
                            #reasoning_effort='high',
                            timeout=None
                            )
                    try:
                        new_content = resp.choices[0].message.content.strip()
                        break
                    except:
                        print('ERROR RECEIVING RESPONSE!')
                        
                # Extract from ``` block if present
                if new_content.count("```") >= 2:
                    start = new_content.find("```")
                    end = new_content.rfind("```")
                    if start != -1 and end != -1 and end > start:
                        # Skip language tag like ```json
                        start = start + 3
                        newline = new_content.find("\n", start)
                        if newline != -1:
                            start = newline + 1
                        new_content = new_content[start:end].strip()

                updated_components[component] = new_content

            except Exception as e:
                print(f"Error generating new {component}: {str(e)}")
                print(traceback.format_exc())
                score = 0

        return updated_components


    def evaluate(
        self,
        batch: list[TestGenDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[TestGenTrajectory, TestGenRolloutOutput]:

        outputs: list[TestGenRolloutOutput] = []
        scores: list[float] = []
        trajectories: list[TestGenTrajectory] | None = [] if capture_traces else None

        if not candidate:
            raise ValueError("Candidate must contain at least one component text.")

        system_prompt = candidate["system_prompt"]   # contains {question}
        rubric_prompt = candidate["rubric_prompt"]    # instructions that GEPA will optimize
        
        openai_requests = []

        for data in batch:
            system_content = system_prompt.format(question=data["question"])
            user_content = rubric_prompt
            openai_requests.append([
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ])

        def fetch_openai(messages):
            for attempt in range(self.retries):
                resp = self.base_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.6,
                        #reasoning_effort="high",
                        timeout=None
                        )
                try:
                    content = resp.choices[0].message.content.strip()
                    return content
                except:
                    print('ERROR RECEIVING RESPONSE!')
                    return ""
        
        try:
            print(f"calling llm with {len(batch)} requests")

            responses = []
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = [executor.submit(fetch_openai, req) for req in openai_requests]
                # Wrap as_completed with tqdm for progress bar
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing requests"):
                    responses.append(future.result())
            print('received responses')
        except Exception as e:
            print('error in getting response')
            raise RuntimeError(f"OpenAI batch_completion failed: {e}") from e
        
        for data, response in zip(batch, responses, strict=False):
            print('#' * 100)
            raw_resp = response

            gold_passed_all = False
            incorrect_failed_some = False
            error_trace = ''
            generated_tests = None
            results_gold = []

            try:
                # print('raw_resp:')
                # print(repr(raw_resp))
                generated_tests = self._extract_json_block(raw_resp)
                # print('generated_tests:')
                # print(generated_tests)
            except Exception:
                error_trace = traceback.format_exc()
                print('Could not extract json block error outside:')
                print(error_trace)
                # print(repr(raw_resp))
                score = 0

            
            if generated_tests is not None:
                try:
                    # print('trying to run gold code on generated tests')
                    fn_name = None if data['fn_name'] == "" else data['fn_name']
                    # print('gold_code:')
                    # print(repr(data["gold_code"]))
                    # print('inputs:')
                    # print(repr(generated_tests["inputs"]))
                    # print('outputs:')
                    # print(repr(generated_tests["outputs"]))
                    success_rate_gold, _ = calculate_score(data["gold_code"], generated_tests["inputs"], generated_tests["outputs"], fn_name=fn_name)
                    print('success_rate_gold: ', success_rate_gold)
                    # print('trying to run incorrect code on generated tests')
                    success_rate_incorrect, _ = calculate_score(data["incorrect_code"], generated_tests["inputs"], generated_tests["outputs"], fn_name=fn_name)
                    print('success_rate_incorrect: ', success_rate_incorrect)

                    if success_rate_gold == 1 and success_rate_incorrect == 0:
                        score = 3.0
                    elif success_rate_gold == 1 and success_rate_incorrect > 0:
                        score = 2.0
                    else:
                        score = 1.0

                except Exception:
                    error_trace = traceback.format_exc()
                    print('Issue when calling run_test():')
                    print(error_trace)
                    score = 0
            else:
                score = 0
            
            

            outputs.append({"generated_tests": raw_resp})
            scores.append(score)

            if capture_traces:
                trajectories.append({
                "data": data,
                "full_assistant_response": raw_resp,
                "generated_tests": generated_tests,
                "gold_passed_all": gold_passed_all,
                "incorrect_failed_some": incorrect_failed_some,
                "error_trace": error_trace,
            })

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[TestGenTrajectory, TestGenRolloutOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        ret_d: dict[str, list[dict[str, Any]]] = {}

        assert len(components_to_update) == 1
        comp = components_to_update[0]

        if eval_batch.trajectories is None:
            raise ValueError("Trajectories required (capture_traces=True).")

        items: list[dict[str, Any]] = []

        for trace_instance in list(zip(eval_batch.trajectories, eval_batch.scores, eval_batch.outputs, strict=False)):
            traj, score, _ = trace_instance
            data = traj["data"]
            generated_outputs = traj["full_assistant_response"]
            gold_passed_all = traj.get("gold_passed_all", None)
            incorrect_failed_some = traj.get("incorrect_failed_some", None)
            error_trace = traj.get("error_trace", '')

            if score == 0:
                feedback = f"There was a formatting/runtime issue in the generated tests. Below is the error traceback: {error_trace}"

            elif score == 1:
                feedback = "The generated test cases were valid but not all of them passed for the correct solution! Make your tests better!"

            elif score == 2:
                feedback = "None of the generated test cases failed for the incorrect solution! Make your tests such that they fail for a slightly incorrect code!"

            else:
                feedback = "The generated test cases are good! They all pass for the gold solution and at least one fails for the incorrect solution."

            items.append(
                {
                    "Inputs": {
                        "question": data["question"],
                        # "gold_code": data["gold_code"],
                        # "incorrect_code": data["incorrect_code"],
                        # "fn_name": data['fn_name']
                    },
                    "Generated Outputs": generated_outputs,
                    "Feedback": feedback,
                })

        ret_d[comp] = items

        if len(items) == 0:
            raise Exception("No valid predictions found for any module.")

        return ret_d
