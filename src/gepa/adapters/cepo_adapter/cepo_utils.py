import re
import time

from dataclasses import dataclass
from collections import Counter
from typing import Literal, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, BadRequestError, InternalServerError


@dataclass
class CepoSimpleConfig:
    planning_n: int  # number of plans generated in planning stage
    planning_m: int  # number of attempts to generate n plans in planning stage
    planning_temperature_step1: float  # temperature for generator in step 1 of planning stage
    planning_temperature_step2: float  # temperature for generator in step 2 of planning stage
    planning_temperature_step3: float  # temperature for generator in step 3 of planning stage
    planning_temperature_step4: float  # temperature for generator in step 4 of planning stage
    planning_max_tokens_step1: int  # maximum number of tokens in step 1 of planning stage
    planning_max_tokens_step2: int  # maximum number of tokens in step 2 of planning stage
    planning_max_tokens_step3: int  # maximum number of tokens in step 3 of planning stage
    planning_max_tokens_step4: int  # maximum number of tokens in step 4 of planning stage



def cepo_simple(system_prompt: str, 
                planning_prompt,
                execution_prompt,
                reflection_prompt,
                question: str, 
                client: Any, 
                model: str, 
                cepo_config: CepoSimpleConfig) -> Tuple[str, int, dict]:
    plans, executions = [], []

    def generate_single_plan(i):
        full_planning_prompt = planning_prompt + f" Here is the question:\n{question}\nRead the question again:\n\n{question}"

        messages = [{"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": full_planning_prompt}]
        response, finish_reason, _ = llm_call_reason_effort_fallback(
            messages=messages,
            client=client,
            model=model,
            max_tokens=cepo_config.planning_max_tokens_step1,
            temperature=cepo_config.planning_temperature_step1,
            top_p=1.0,
            reasoning_effort_levels=["high", "medium"]
        )
        

        if finish_reason == "length":
            return i, None, None
        parsed_plan = response


        # Step 2 – Execute plan
        messages.append({"role": "assistant", "content": parsed_plan})
        messages.append({"role": "user", "content": execution_prompt})
        response, finish_reason, _ = llm_call_reason_effort_fallback(
                messages=messages,
                client=client,
                model=model,
                max_tokens=cepo_config.planning_max_tokens_step2,
                temperature=cepo_config.planning_temperature_step2,
                top_p=1.0,
                reasoning_effort_levels=["high", "medium"]
            )

        if finish_reason == "length":
            return i, None, None

        parsed_exec = response
        return i, parsed_plan, parsed_exec 

    # Step 1 & 2: Parallel planning + execution
    with ThreadPoolExecutor(max_workers=cepo_config.planning_m) as executor:
        futures = [executor.submit(generate_single_plan, i) for i in range(cepo_config.planning_m)]

        for future in as_completed(futures):
            i, plan, execution = future.result()
            if plan and execution:
                plans.append((i, plan))
                executions.append((i, execution))
            if len(plans) == cepo_config.planning_n:
                break

    plans = [plan for _, plan in sorted(plans)]  # keep original order
    executions = [execution for _, execution in sorted(executions)]
    assert len(plans) == len(executions)

    if not executions:
        # fallback plan
        fallback_generation, fallback_messages = fallback_direct_answer(client, model, question)
        executions.append(fallback_generation)

    # Step 3 - Review and consolidate plans
    executions_message = ""
    for i, execution in enumerate(executions):
        executions_message += f"Response {i + 1}:\n{execution}\n\n"
    executions_message = executions_message.rstrip()

    reflection_prompt = reflection_prompt + f"Here is the question:\n{question} and N = {len(executions)}."

    user_content = f"Previous responses to review:\n\n{executions_message}\n\n{reflection_prompt}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    
    
    response, finish_reason, completion_tokens = llm_call_reason_effort_fallback(
                messages=messages,
                client=client,
                model=model,
                max_tokens=cepo_config.planning_max_tokens_step3,
                temperature=cepo_config.planning_temperature_step3,
                top_p=1.0,
                reasoning_effort_levels=["high", "medium"]
            )
    
    if response is None or finish_reason == "length":
        print("Step 3 failed and only taking plans[0]")
        final_solution = plans[0]
    else:
        final_solution = response

    # # Step 4 – Final answer
    # content = f"Use your final solution from above to correctly answer the question. Here is the question:\n{question}"
    # messages = [
    #     {"role": "system", "content": system_prompt}, 
    #     {"role": "user", "content": f"Here's my final solution: {final_solution}\n\nNow {content}"}
    # ]
    # response, finish_reason, _ = llm_call_reason_effort_fallback(
    #         messages=messages,
    #         client=client,
    #         model=model,
    #         max_tokens=cepo_config.planning_max_tokens_step4,
    #         temperature=cepo_config.planning_temperature_step4,
    #         top_p=1.0,
    #         reasoning_effort_levels=["medium", "low"]
    #     )
    # if response is None or finish_reason == "length":
    #     print("Step 4 failed and only taking step 3 output")
    #     final_output = final_solution
    # else:
    #     final_output = response
    final_output = final_solution

    return final_output, plans, executions 


 

def fallback_direct_answer(question, client, model, max_tokens=None, temperature=1.0, top_p=1.0):
    messages = [
        {"role": "user", "content": question},
    ]

    response, finish_reason, completion_tokens = llm_call_reason_effort_fallback(
                messages=messages,
                client=client,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                reasoning_effort_levels=["high", "medium", "low"]
            )
    if response is None or finish_reason == "length":
        print("direct answer problem")
        response = ""
    messages.append({"role": "assistant", "content": response})
    return response, messages

def extract_llm_response(response):
    # Case 1: non-streaming response (dict-like object)
    if hasattr(response, "choices") and hasattr(response.choices[0], "message"):
        content = response.choices[0].message.content
        if content:
            content = content.strip()
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        return content, finish_reason

    # Case 2: streaming response (generator)
    full_content = ""
    finish_reason = None
    for chunk in response:
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            full_content += delta.content
        if chunk.choices[0].finish_reason is not None:
            finish_reason = chunk.choices[0].finish_reason
    return full_content.strip(), finish_reason


def remove_think_section(response):
    if not isinstance(response, str) or not response:
        return ""
    if not response.startswith("<think>") and "<think>" not in response:
        return response
    match = re.search(r"</think>\s*(.*)", response, re.DOTALL)
    if match:
        parsed_response = match.group(1)
        return parsed_response
    else:
        return response


def llm_call(messages, client, model, max_tokens, temperature, top_p, reasoning_effort):
    tries = 2  # 1 initial + 2 retries
    for attempt in range(tries):
        try:
            response_object = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                top_p=top_p,
                reasoning_effort=reasoning_effort,
            )
            response_text, finish_reason = extract_llm_response(response_object)
            completion_tokens = getattr(getattr(response_object, "usage", None), "completion_tokens", 0) or 0
            # Normalize None → ""
            response_text = response_text or ""
            if response_text is not None:
                response_text = remove_think_section(response_text)
            return response_text, finish_reason, completion_tokens

        except (BadRequestError, InternalServerError) as e:
            # Retry on 400 or 500
            if attempt < tries - 1:
                sleep_time = 0.2 * (attempt + 1)
                print(f"Got {e.__class__.__name__}, retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                continue
            raise


def llm_call_reason_effort_fallback(messages, client, model, max_tokens, temperature, top_p, reasoning_effort_levels):
    response = None
    finish_reason = "error"
    completion_tokens = 0
    # There are two types of error to handle
    # Case 1: the model didn't finish generation, 
    # there will still be a response but with content to None;
    # we just lower the reasoning effort and see

    # Case 2: gpt-oss's "expected output number" error
    # this will usually trigger a 400 http error and cannot be recovered
    # our only option is to retry and then lower reasoning effort
    # this seems a transient error but no solution yet (https://github.com/pydantic/pydantic-ai/issues/2449)
    # We might need to modify vllm/gpt-oss source code for this...
    for effort in reasoning_effort_levels:
        try:
            response, finish_reason, completion_tokens = llm_call(
                messages=messages,
                client=client,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                reasoning_effort=effort,
            )
            # This is likely a context issue
            if response is not None and finish_reason != "length":
                return response, finish_reason, completion_tokens
            # print("Reasoning fallback from", effort, "to lower ones")
        except (BadRequestError, InternalServerError) as e:
            # After 2 retries at this effort failed with 400 → degrade
            print("400/500 persisted after retries at reasoning effort", effort, "→ degrading")
            continue

    # if nothing worked, just return none
    return response, finish_reason, completion_tokens