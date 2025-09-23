# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa


from gepa.proposer.reflective_mutation.base import Signature


class InstructionProposalSignature(Signature):
#     prompt_template = """I provided an assistant with the following instructions to perform a task for me:
# ```
# <curr_instructions>
# ```

# The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
# ```
# <inputs_outputs_feedback>
# ```

# Your task is to write a new instruction for the assistant.

# Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

# Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

# Provide the new instructions within ``` blocks."""

    prompt_template = """I provided an assistant with the following {PROMPT_TYPE} instruction:
```
<curr_instructions>
```

The following are examples of different task inputs, the assistant’s responses, and feedback on how the {PROMPT_TYPE} could be improved:
```
<inputs_outputs_feedback>
```

Your task is to propose a revised {PROMPT_TYPE} instruction for the assistant.

Guidelines {GUIDELINES}

Return only the new instruction inside ``` blocks."""

    input_keys = ["current_instruction_doc", "dataset_with_feedback", "prompt_type"]
    output_keys = ["new_instruction"]

    @classmethod
    def get_guidelines(cls, prompt_type):
        guidelines = {
            "PLANNING": "- Focus ONLY on improving the *structure and clarity of planning* (e.g., require decomposition into steps, explicit input/output contracts, consideration of boundary cases, and a self-check).\n- DO NOT include any dataset-specific content, numeric constraints, input/output formats, examples, or code.\n- Keep the instruction general so it can apply to many tasks.\n- The revision should be concise, clear, and in natural language.",
            
            "EXECUTION": "- Focus ONLY on improving how the assistant should *execute a plan* (e.g., methodical implementation, validation of intermediate steps, error checking).\n- Emphasize careful execution especially for steps where confidence is lower.\n- DO NOT include any dataset-specific content, numeric constraints, input/output formats, examples, or code.\n- Keep the instruction general so it can apply to many tasks.\n- The revision should be concise, clear, and in natural language.",
            
            "REFLECTION": "- Focus ONLY on improving how the assistant should *reflect on and validate* their work (e.g., reviewing for inconsistencies, verifying solutions, refining approaches).\n- Emphasize careful review of the entire problem-solving process.\n- DO NOT include any dataset-specific content, numeric constraints, input/output formats, examples, or code.\n- Keep the instruction general so it can apply to many tasks.\n- The revision should be concise, clear, and in natural language."
        }
        return guidelines.get(prompt_type, guidelines["PLANNING"])

    @classmethod
    def prompt_renderer(cls, input_dict: dict[str, str]) -> str:
        def format_samples(samples):
            def render_value(value, level=3):
                # level controls markdown header depth (###, ####, etc.)
                if isinstance(value, dict):
                    s = ""
                    for k, v in value.items():
                        s += f"{'#' * level} {k}\n"
                        s += render_value(v, min(level + 1, 6))
                    if not value:
                        s += "\n"
                    return s
                elif isinstance(value, (list, tuple)):
                    s = ""
                    for i, item in enumerate(value):
                        s += f"{'#' * level} Item {i + 1}\n"
                        s += render_value(item, min(level + 1, 6))
                    if not value:
                        s += "\n"
                    return s
                else:
                    return f"{str(value).strip()}\n\n"

            def convert_sample_to_markdown(sample, examplenum):
                s = f"# Example {examplenum}\n"
                for key, val in sample.items():
                    s += f"## {key}\n"
                    s += render_value(val, level=3)
                return s

            return "\n\n".join(convert_sample_to_markdown(sample, i + 1) for i, sample in enumerate(samples))
        
        prompt_type = input_dict.get("prompt_type", "PLANNING").upper()
        guidelines = cls.get_guidelines(prompt_type)

        prompt = cls.prompt_template
        prompt = prompt.replace("{PROMPT_TYPE}", prompt_type)
        prompt = prompt.replace("{GUIDELINES}", guidelines)
        prompt = prompt.replace("<curr_instructions>", input_dict["current_instruction_doc"])
        prompt = prompt.replace("<inputs_outputs_feedback>", format_samples(input_dict["dataset_with_feedback"]))
        
        return prompt

    @classmethod
    def output_extractor(cls, lm_out: str) -> dict[str, str]:
        # Extract ``` blocks
        new_instruction = None
        if lm_out.count("```") >= 2:
            start = lm_out.find("```")
            end = lm_out.rfind("```")
            if start >= end:
                new_instruction = lm_out
            if start == -1 or end == -1:
                new_instruction = lm_out
            else:
                new_instruction = lm_out[start+3:end].strip()
        else:
            lm_out = lm_out.strip()
            if lm_out.startswith("```"):
                lm_out = lm_out[3:]
            if lm_out.endswith("```"):
                lm_out = lm_out[:-3]
            new_instruction = lm_out

        return {"new_instruction": new_instruction}
