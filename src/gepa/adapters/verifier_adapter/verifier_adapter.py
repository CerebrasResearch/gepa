# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from typing import Any, Callable, TypedDict

from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from .prompts import SYSTEM_PROMPT_DEEPSEEK_GRM, USER_PROMPT_DEEPSEEK_GRM_2_RESPONSES_SHORTENED
import litellm
import re

# litellm._turn_on_debug()


def render_user_prompt(tpl: str, *, question: str, response1: str, response2: str) -> str:
    """
    Escape all curly braces in tpl, except our placeholders {question}, {response1}, {response2},
    then format with the provided values.
    """
    # Temporarily mask intended placeholders so they survive escaping
    ph = {
        "{question}":  "§§PH_QUESTION§§",
        "{response1}": "§§PH_RESPONSE1§§",
        "{response2}": "§§PH_RESPONSE2§§",
    }
    tmp = tpl
    for k, v in ph.items():
        tmp = tmp.replace(k, v)

    # Escape every remaining brace so LaTeX/markdown stay literal
    tmp = tmp.replace("{", "{{").replace("}", "}}")

    # Restore placeholders and format
    for k, v in ph.items():
        tmp = tmp.replace(v, k)

    return tmp.format(question=question, response1=response1, response2=response2)


def strip_first_last_instruction(prompt: str) -> str:
    """
    Remove the first and last non‑empty base instructions from a multiline problem
    """
    lines = [ln for ln in prompt.splitlines() if ln.strip()]
    return "\n".join(lines[1:-1])

# DataInst, Trajectory, RolloutOutput
class DefaultDataInst(TypedDict):
    input: str
    additional_context: dict[str, str]
    answer: str

class DefaultTrajectory(TypedDict):
    data: DefaultDataInst
    full_assistant_response: str

class DefaultRolloutOutput(TypedDict):
    full_assistant_response: str

class VerifierAdapter(GEPAAdapter[DefaultDataInst, DefaultTrajectory, DefaultRolloutOutput]):
    def __init__(
        self,
        model: str,
        api_base: str | None = "http://localhost:8000",
        max_litellm_workers: int = 10,
        reflection_model: str | None = None,
        reflection_api_base:  str | None = "http://127.0.0.1:8190/v1" ) -> None:
        
        self.model = model
        self.litellm = litellm
        self.max_litellm_workers = max_litellm_workers
        self.api_base = api_base
        if self.api_base is None or self.api_base == "":
            self.api_base = None
        self.reflection_api_base = reflection_api_base
        self.reflection_model = reflection_model


    def extract_rating(self, solution_str):
        # match = re.search(r"Rating:\s*\[\[(\d+)\]\]", solution_str)
        match = re.search(r'\\boxed\{(.*?)\}', solution_str[-30:], re.DOTALL)
        if match:
            try:
                boxed_content = match.group(1)
                rating = list(map(int, re.findall(r'\d+', boxed_content)))
            except:
                rating = [-1, -1]
        else:
            rating = [-1, -1]
        if len(rating) != 2:
            rating = [-1, -1]
        return rating

    def post_process_output(self, output: str, answer: list) :
        # extracts the ratings and the calculates the final score based on the ratings
        extracted_rating = self.extract_rating(output)
        gt_rating = answer # remapped gt
        
        accuracy = 1
        accuracy_of_each_response = [False, False]

        for pred_val, gt_val in zip(extracted_rating, gt_rating):
            if pred_val == -1:
                accuracy = 0
            pred_label = int(pred_val > 5)
            accuracy &= (pred_label == gt_val)
            accuracy_of_each_response.append(int(pred_label == gt_val))

        return accuracy, accuracy_of_each_response

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """
        
        """
        updated_components = {}
        for component in components_to_update:
            print('comopnent:', component)
            print('\n')

            # Get examples for this component (where we have different components for the system prompt and user prompt(s))
            examples = reflective_dataset.get(component, [])
            if not examples:
                continue
            
            # Format examples into readable strings for the reflection prompt
            examples_formatted = ""
            for i, example in enumerate(examples):
                examples_formatted += f"Example {i+1}:\n"
                examples_formatted += f"Formatted Prompt Sent to Model: {example['Inputs']}\n"
                examples_formatted += f"Outputs: {example['Generated Outputs']}\n"
                examples_formatted += f"Feedback: {example['Feedback']}\n\n"

            if component == "user_prompt_template":
                reflection_prompt = f"""
I provided the assistant with the following user prompt template to evaluate responses for me:
```
{candidate['user_prompt_template']}
``` 
The following are examples where this template was filled with specific questions and responses, then sent to the model.
Each example shows the fully formatted prompt that was sent to the model, the model's response, and feedback on that response:
```
{examples_formatted}
```
Your task is to write a new user prompt template for the assistant. 
Read the formatted prompts carefully to understand how the template is being used, and identify what the assistant is being asked to do.
The assistant needs to evaluate and compare responses, then provide ratings in a specific format.
Read all the model responses and the corresponding feedback. The assistant should follow a clear step-by-step evaluation process and provide ratings in a consistent format with \\boxed{{x, x}} at the end.
**IMPORTANT**: The new user prompt template MUST retain these exact placeholders: {{question}}, {{response1}} and {{response2}}. These placeholders are essential for the prompt to function correctly.
Provide the new user prompt template within ``` blocks.
"""
            else: # system_prompt
                reflection_prompt = f"""I provided an assistant with the following system prompt to evaluate responses:
```
{candidate['system_prompt']}
```
The following are examples where this system prompt was used along with a user prompt to generate responses.
Each example shows the fully formatted user prompt that was sent to the model, the model's response, and feedback on that response:
```
{examples_formatted}
```
Your task is to write a new system prompt for the assistant.
Read the formatted prompts and model outputs carefully to understand how the system prompt should guide the model's evaluation of responses.
The system prompt should set clear expectations for how the assistant should evaluate and compare responses.
Read all the model responses and the corresponding feedback. The assistant should provide consistent, accurate ratings and format them properly with \\boxed{{x, x}} notation.
Provide the new system prompt within ``` blocks."""
            try:
                print("=== Reflection Prompt Start ===")
                print(reflection_prompt)
                print("=== Reflection Prompt End ===")

                response = self.litellm.completion(
                    model=self.reflection_model,
                    api_base=self.reflection_api_base, 
                    api_key="EMPTY",
                    messages=[{"role": "user", "content": reflection_prompt}]
                )
                new_content = response.choices[0].message['content'].strip()
                print ("=== Reflection Response Start ===")
                print(new_content)
                print ("=== Reflection Response End ===")
            
                # Extract content from code blocks
                if new_content.count("```") >= 2:
                    start = new_content.find("```")
                    end = new_content.rfind("```")
                    if start != -1 and end != -1 and end > start:
                        start = start + 3
                        start = start + new_content[start:].find("\n") + 1
                        new_content = new_content[start:end].strip()
                
                # Validate user_prompt_template has required placeholders
                if component == "user_prompt_template":
                    required_placeholders = ["{question}", "{response1}", "{response2}"]
                    if all(placeholder in new_content for placeholder in required_placeholders):
                        updated_components[component] = new_content
                    else:
                        print(f"Warning: Generated template missing required placeholders. Not updating.")
                else:
                    print('Using default component (no updates)')
                    updated_components[component] = new_content
                    
            except Exception as e:
                print(f"Error generating new {component}: {str(e)}")
    
        return updated_components          


    def evaluate(
        self,
        batch: list[DefaultDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[DefaultTrajectory, DefaultRolloutOutput]:
        outputs: list[DefaultRolloutOutput] = []
        scores: list[float] = []
        trajectories: list[DefaultTrajectory] | None = [] if capture_traces else None
        # Here, we assume that `candidate` contains keys "system_prompt" and "user_prompt_template" with the following structure:
        # {'user_prompt_template': USER_PROMPT_DEEPSEEK_GRM_2_RESPONSES_SHORTENED, "system_prompt": "<system prompt>"

        system_content = candidate.get("system_prompt", SYSTEM_PROMPT_DEEPSEEK_GRM)
        user_prompt_template = candidate.get("user_prompt_template", USER_PROMPT_DEEPSEEK_GRM_2_RESPONSES_SHORTENED)

        litellm_requests = []

        for data in batch:

            question = data['input']
            other_data = data['additional_context']

            messages = [{
                "role": "system",
                "content": system_content
            },{
                "role": "user",
                "content": render_user_prompt(
                    user_prompt_template,
                    question=question,
                    response1=other_data['response1'],
                    response2=other_data['response2'],
                ),
            }]

            # print("-----messages(base)----:", messages)
            litellm_requests.append(messages)

        try:
            if isinstance(self.model, str):
                responses = self.litellm.batch_completion(model=self.model, messages=litellm_requests, max_workers=self.max_litellm_workers, api_base=self.api_base, api_key='EMPTY', drop_params=True, stream=False)
                responses = [resp.choices[0].message['content'].strip() for resp in responses]
                # dumpy responses in jsonl

            else:
                responses = [self.model(messages) for messages in litellm_requests]
        except Exception as e:
            raise e

        for data, assistant_response in zip(batch, responses, strict=False):
            
            output = {"full_assistant_response": assistant_response}
            # print("\n-----full assistant response(base)----\n:", assistant_response)

            # N: change scoring from the model responses
            score, acc_per_response = self.post_process_output(assistant_response, data['additional_context']['answer'])

            outputs.append(output)
            scores.append(score)

            if capture_traces:
                trajectories.append(
                    {
                        "data": data,
                        "full_assistant_response": assistant_response,
                        "accuracy_of_each_response": acc_per_response,
                    }
                )

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[DefaultTrajectory, DefaultRolloutOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        ret_d: dict[str, list[dict[str, Any]]] = {}

        assert len(components_to_update) == 1
        comp = components_to_update[0]

        items: list[dict[str, Any]] = []
        trace_instances = list(zip(eval_batch.trajectories, eval_batch.scores, eval_batch.outputs, strict=False))

        for trace_instance in trace_instances:
            traj, score, _ = trace_instance
            data = traj["data"]
            generated_outputs = traj["full_assistant_response"]
            acc_per_response = traj.get("accuracy_of_each_response", [-1, -1])

            # if score > 0.0:
            #     feedback = f"The generated response is correct.'"
            # else:
            #     additional_context_str = "\n".join(f"{k}: {v}" for k, v in data["additional_context"].items())
            #     feedback = f"The generated response is incorrect."
            feedback = ""
            if acc_per_response[0] == 1:
                feedback += f" The rating for response 1 is correct."
            else:
                feedback += f" The rating for response 1 is incorrect."

            if acc_per_response[1] == 1:
                feedback += f" The rating for response 2 is correct."
            else:
                feedback += f" The rating for response 2 is incorrect."
            
            if acc_per_response[0] == 0 or acc_per_response[1] == 0:
                feedback+= "One of the ratings doesn't match the expectation. Overall, both ratings should be accurate and reflect the quality of the responses. "
            
            if acc_per_response[0] == 1 and acc_per_response[1] == 1:
                feedback+= "Overall, both ratings are accurate and reflect the quality of the responses. "



            formatted_input = render_user_prompt(
                USER_PROMPT_DEEPSEEK_GRM_2_RESPONSES_SHORTENED,
                question=data['input'],
                response1=data['additional_context']['response1'],
                response2=data['additional_context']['response2'],
            )

            d = {
                "Inputs": formatted_input,
                "Generated Outputs": generated_outputs,
                "Feedback": feedback,
            }

            items.append(d)

        ret_d[comp] = items

        if len(items) == 0:
            raise Exception("No valid predictions found for any module.")

        return ret_d