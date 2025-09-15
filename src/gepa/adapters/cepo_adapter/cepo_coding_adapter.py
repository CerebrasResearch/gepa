import logging
import random
from typing import Any, TypedDict
from pydantic import BaseModel, Field
from openai import OpenAI
from tqdm import tqdm
from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from gepa.adapters.cepo_adapter.cepo_utils import CepoSimpleConfig, cepo_simple
from gepa.adapters.cepo_adapter.coding_utils import calculate_score, extract_code


# TODO:
# a fully functional flow on success rate only, test on a few examples
    # Make reflection function work, 
    # make training script work
    # prepare some examples from HF hard, and trigger a run
# integrate debugging information
# integrate intermediate plans executions inconsistency to trajectory
# Integrate optional ground truth code


global_candidate = {
    "cepo_planning_prompt": "To answer this question, can you come up with a concise plan to solve it step-by-step but do not provide the final answer. Also, for each step, provide your confidence in the correctness of that step as well as your ability to execute it correctly. ",
    "cepo_execution_prompt": "Can you execute the above plan step-by-step to produce the final answer. Be extra careful when executing steps where your confidence is lower.",
    "cepo_reflection_prompt": "Can you review your last N responses and identify any inconsistency between them. After that, can you address it and present a final step-by-step solution to the problem?",
}



class CepoCodingDataInst(TypedDict):
    """
    User-defined type of input data to the program under optimization.
    """
    question: str # Input coding question
    test_inputs: list[str] 
    test_outputs: list[str]
    fn_name: str


class CepoCodingTrajectory(TypedDict):
    data: CepoCodingDataInst # The input data instance
    full_assistant_response: str # The full raw response from the model
    debug_results: list[dict] # A list of {"test_input": ..., "predicted_output": ..., "ground_truth_output": ..., "status": ...} for each test case


class CepoCodingRolloutOutput(TypedDict):
    full_assistant_response: str

class CepoCodingStructuredOutput(BaseModel):
    final_code: str = Field(..., description="The final extracted code only")
    final_solution: str = Field(..., description="Step-by-step reasoning or explanation that led to the answer.")

class CepoCodingAdapter(GEPAAdapter[CepoCodingDataInst, CepoCodingTrajectory, CepoCodingRolloutOutput]):
    """
    CepoCoding Adapter is a GEPAAdapter for any dataset that contains coding question and test cases.
    """

    def __init__(
        self,
        model: str,
        failure_score: float = 0.0,
        api_base: str | None = "http://localhost:8190/v1",
    ) -> None:

        self.model = model
        self.failure_score = failure_score
        self.log = logging.getLogger("CepoCodingAdapter")

        self.client = OpenAI(
            api_key="serving-on-vllm",
            base_url=api_base,
            timeout=None,
            max_retries=0,
        )
        self.cepo_config = CepoSimpleConfig(
            planning_n=2,
            planning_m=3,
            planning_temperature_step1=1.0,
            planning_temperature_step2=1.0,
            planning_temperature_step3=1.0,
            planning_temperature_step4=0.5,
            planning_max_tokens_step1=40960,
            planning_max_tokens_step2=40960,
            planning_max_tokens_step3=40960,
            planning_max_tokens_step4=40960,
        )


    def evaluate(
        self,
        batch: list[CepoCodingDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[CepoCodingTrajectory, CepoCodingRolloutOutput]:
        
        outputs: list[CepoCodingRolloutOutput] = []
        scores: list[float] = []
        trajectories: list[CepoCodingTrajectory] | None = [] if capture_traces else None
        
        if not candidate:
            raise ValueError("Candidate must contain at least one component text.")

        for data in tqdm(batch, desc="Evaluation..."):
            # Step 1: use cepo simple (bon = 1, planning_n=2) to get a step 4 solution
            final_output, plans, executions = cepo_simple(
                system_prompt="",
                planning_prompt=candidate["cepo_planning_prompt"],
                execution_prompt=global_candidate["cepo_execution_prompt"],
                reflection_prompt=global_candidate["cepo_reflection_prompt"],
                question=data["question"],
                client=self.client,
                model=self.model,
                cepo_config=self.cepo_config,

            )

            outputs.append(
                {"full_assistant_response": final_output}
            )

            # Step 2: Check correctness and only build trajectory on success rate
            extracted_code = extract_code(final_output)

            success_rate, debug_results = calculate_score(extracted_code, 
                                    data["test_inputs"],
                                    data["test_outputs"],
                                    data["fn_name"])
            scores.append(success_rate)


            # TODO: Might need to add more
            if capture_traces:
                trajectories.append(
                    {"data": data, 
                     "full_assistant_response": final_output,
                     "debug_results": debug_results,
                    }
            )

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[CepoCodingTrajectory, CepoCodingRolloutOutput],
        components_to_update: list[str],
        max_cases: int = 15,
    ) -> dict[str, list[dict[str, Any]]]:
        # TODO: Figure out what is this component name and why we have such assert
        assert len(components_to_update) == 1
        comp = components_to_update[0]

        if eval_batch.trajectories is None:
            raise ValueError("Trajectories required (capture_traces=True).")

        items: list[dict[str, Any]] = []
        for traj, score in tqdm(
            zip(eval_batch.trajectories, eval_batch.scores, strict=False),
            total=len(eval_batch.trajectories),
            desc="Reflection..."
        ):
            data = traj["data"]
            generated_outputs = traj["full_assistant_response"]
            debug_results = traj["debug_results"]
            
            # TODO: add reasonable threashold for code passing rate
            if score == 1.0:
                feedback = f"The generated code is correct and it passed all test cases."
            else:
                failed = [r for r in (debug_results or []) if r.get("status") != "passed"]
                if len(failed) > max_cases:
                    failed = random.sample(failed, max_cases)
                lines = []
                for i, r in enumerate(failed, 1):
                    lines.append(
                        f"- Case {i} | status={r.get('status')}\n"
                        f"  input: {r.get('test_input')}\n"
                        f"  expected: {r.get('gt_output')}\n"
                        f"  predicted: {r.get('pred_output')}"
                    )
                feedback = (
                    "The generated code is incorrect.\n"
                    "Here are failing test cases:\n"
                    + "\n".join(lines)
                    + "\n\nPlease analyze why these failures occur and propose a minimal fix."
                )


            items.append(
                {
                    "Inputs": data["question"],
                    "Generated Outputs": generated_outputs,
                    "Feedback": feedback,
                }
            )

        if len(items) == 0:
            raise Exception("No valid predictions found for any module.")

        return {comp: items}