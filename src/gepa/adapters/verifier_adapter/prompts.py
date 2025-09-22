

SYSTEM_PROMPT_DEEPSEEK_GRM = """You are a skilled little expert at scoring responses. You should evaluate given responses based \
    on the given judging criteria.""" 

USER_PROMPT_DEEPSEEK_GRM_2_RESPONSES_SHORTENED = """Given a question and multiple responses from the Assistan score the responses. Feel free to state \
potential other specific criteria to the query, the weights of different criteria, and then provide an overall comprehensive score \
upon them.\n Each score is an integer between 1 and 10, with a higher score indicating that the response meets the relevant \
criteria more closely. For example, a score of 1 means the response does not meet the criteria at all, a score of 6 means \
the response meets only some parts, and a score of 10 means the response perfectly meets the evaluation criteria.\n \
Before scoring, please analyze step by step. Your scoring needs to be as strict as possible. \
#### Question Begin ####\n{question}\n \
#### Responses to be Scored #### \
[The Begin of Response 1]\n{response1}\n[The End of Response 1]\n \
[The Begin of Response 2]\n{response2}\n[The End of Response 2]\n 
#### Output Format Requirements #### \
Output with three lines \
Specific Criteria: <Other potential criteria specific to the query and the context, and the \
weights of each criteria>. \
Analysis: <Compare different responses based on given Criteria>. \
Scores: <the overall comprehensive score of all responses in order, separate by comma in the \
boxed, e.g., \\boxed{{x, x}}>. \
"""





USER_PROMPT_DEEPSEEK_GRM_2_RESPONSES = """Given a question and multiple responses from the Assistant, you need to refer to the \
[General Evaluation Criteria] to score the responses. Based on the general evaluation criteria, state \
potential other specific criteria to the query, the weights of different criteria, and then provide an overall comprehensive score \
upon them.\n Each score is an integer between 1 and 10, with a higher score indicating that the response meets the relevant \
criteria more closely. For example, a score of 1 means the response does not meet the criteria at all, a score of 6 means \
the response meets only some parts, and a score of 10 means the response perfectly meets the evaluation criteria.\n \
Before scoring, please analyze step by step. Your scoring needs to be as strict as possible. \
#### Evaluation Criteria ####
1. Correctness: \n - Correct Final response (10 points): The final answer is completely correct, with no errors or omissions.\n \
- Mostly Correct (6-9 points): The final answer is mostly correct, with minor errors or omissions that do not significantly affect the overall correctness.\n \
- Partially Correct (3-5 points): The final answer is partially correct, with significant errors or omissions that affect the overall correctness.\n \
- Incorrect (1-2 points): The final answer is completely incorrect, with no correct information.\n \
Example: If the question is about solving a mathematical problem and the response provides the final correct answer, it falls under "Correct Final response." \
2. Instruction Adherence:\n - Fully Adhered (9-10 points): The response fully complies with all instructions and requirements of the question.\n \
- Partially Adhered (6-8 points): The response meets most of the instructions but has some omissions or misunderstandings.\n\
- Basically Adhered (3-5 points): The response meets some instructions, but the main requirements are not fulfilled.\n\
- Not Adhered (1-2 points): The response does not meet any instructions.\n\
Example: If the question requires three examples and the response provides only one, it falls under "Partially Adhered." \
3. Usefulness:\n - Highly Useful (9-10 points): The response provides comprehensive and accurate information, fully addressing the issue.\n - Useful but Incomplete (6-8 points): \
The response provides some useful information, but lacks details or accuracy.\n - Limited Usefulness (3-5 points): The response offers little useful information, with most content \
being irrelevant or incorrect.\n - Useless or Incorrect (1-2 points): The response is completely \
irrelevant or incorrect.\n Example: If there are factual errors in the response but the overall \
direction is correct, it falls under "Useful but Incomplete." \
4. Relevance:\n - Highly Relevant (9-10 points): The response is highly relevant to the \
question, with information closely aligned with the topic.\n - Generally Relevant (6-8 points): \
The response is generally relevant but includes some unnecessary information.\n - Partially \
Relevant (3-5 points): The response has a lot of content that deviates from the topic.\n - Not \
Relevant (1-2 points): The response is completely irrelevant.\n Example: If the response strays \
from the topic but still provides some relevant information, it falls under "Partially Relevant." \
#### Question Begin ####\n{question}\n \
#### Responses to be Scored #### \
[The Begin of Response 1]\n{response1}\n[The End of Response 1]\n \
[The Begin of Response 2]\n{response2}\n[The End of Response 2]\n 
#### Output Format Requirements #### \
Output with three lines \
Specific Criteria: <Other potential criteria specific to the query and the context, and the \
weights of each criteria>. \
Analysis: <Compare different responses based on given Criteria>. \
Scores: <the overall comprehensive score of all responses in order, separate by comma in the \
boxed, e.g., \\boxed{{x, x}}>. \
"""



USER_PROMPT_DEEPSEEK_GRM_3_RESPONSES = """Given a question and multiple responses from the Assistant, you need to refer to the \
[General Evaluation Criteria] to score the responses. Based on the general evaluation criteria, state \
potential other specific criteria to the query, the weights of different criteria, and then provide an overall comprehensive score \
upon them.\n Each score is an integer between 1 and 10, with a higher score indicating that the response meets the relevant \
criteria more closely. For example, a score of 1 means the response does not meet the criteria at all, a score of 6 means \
the response meets only some parts, and a score of 10 means the response perfectly meets the evaluation criteria.\n \
Before scoring, please analyze step by step. Your scoring needs to be as strict as possible. \
#### Evaluation Criteria ####
1. Correctness: \n - Correct Final response (10 points): The final answer is completely correct, with no errors or omissions.\n \
- Mostly Correct (6-9 points): The final answer is mostly correct, with minor errors or omissions that do not significantly affect the overall correctness.\n \
- Partially Correct (3-5 points): The final answer is partially correct, with significant errors or omissions that affect the overall correctness.\n \
- Incorrect (1-2 points): The final answer is completely incorrect, with no correct information.\n \
Example: If the question is about solving a mathematical problem and the response provides the final correct answer, it falls under "Correct Final response." \
2. Instruction Adherence:\n - Fully Adhered (9-10 points): The response fully complies with all instructions and requirements of the question.\n \
- Partially Adhered (6-8 points): The response meets most of the instructions but has some omissions or misunderstandings.\n\
- Basically Adhered (3-5 points): The response meets some instructions, but the main requirements are not fulfilled.\n\
- Not Adhered (1-2 points): The response does not meet any instructions.\n\
Example: If the question requires three examples and the response provides only one, it falls under "Partially Adhered." \
3. Usefulness:\n - Highly Useful (9-10 points): The response provides comprehensive and accurate information, fully addressing the issue.\n - Useful but Incomplete (6-8 points): \
The response provides some useful information, but lacks details or accuracy.\n - Limited Usefulness (3-5 points): The response offers little useful information, with most content \
being irrelevant or incorrect.\n - Useless or Incorrect (1-2 points): The response is completely \
irrelevant or incorrect.\n Example: If there are factual errors in the response but the overall \
direction is correct, it falls under "Useful but Incomplete." \
4. Relevance:\n - Highly Relevant (9-10 points): The response is highly relevant to the \
question, with information closely aligned with the topic.\n - Generally Relevant (6-8 points): \
The response is generally relevant but includes some unnecessary information.\n - Partially \
Relevant (3-5 points): The response has a lot of content that deviates from the topic.\n - Not \
Relevant (1-2 points): The response is completely irrelevant.\n Example: If the response strays \
from the topic but still provides some relevant information, it falls under "Partially Relevant." \
#### Question Begin ####\n{question}\n \
#### Responses to be Scored #### \
[The Begin of Response 1]\n{response1}\n[The End of Response 1]\n \
[The Begin of Response 2]\n{response2}\n[The End of Response 2]\n \
[The Begin of Response 3]\n{response3}\n[The End of Response 3]\n 
#### Output Format Requirements #### \
Output with three lines \
Specific Criteria: <Other potential criteria specific to the query and the context, and the \
weights of each criteria>. \
Analysis: <Compare different responses based on given Criteria>. \
Scores: <the overall comprehensive score of all responses in order, separate by comma in the \
boxed, e.g., \\boxed{{x, x, x}}>. \
"""


USER_PROMPT_DEEPSEEK_GRM_5_RESPONSES = """Given a question and multiple responses from the Assistant, you need to refer to the \
[General Evaluation Criteria] to score the responses. Based on the general evaluation criteria, state \
potential other specific criteria to the query, the weights of different criteria, and then provide an overall comprehensive score \
upon them.\n Each score is an integer between 1 and 10, with a higher score indicating that the response meets the relevant \
criteria more closely. For example, a score of 1 means the response does not meet the criteria at all, a score of 6 means \
the response meets only some parts, and a score of 10 means the response perfectly meets the evaluation criteria.\n \
Before scoring, please analyze step by step. Your scoring needs to be as strict as possible. \
#### Evaluation Criteria ####
1. Correctness: \n - Correct Final response (10 points): The final answer is completely correct, with no errors or omissions.\n \
- Mostly Correct (6-9 points): The final answer is mostly correct, with minor errors or omissions that do not significantly affect the overall correctness.\n \
- Partially Correct (3-5 points): The final answer is partially correct, with significant errors or omissions that affect the overall correctness.\n \
- Incorrect (1-2 points): The final answer is completely incorrect, with no correct information.\n \
Example: If the question is about solving a mathematical problem and the response provides the final correct answer, it falls under "Correct Final response." \
2. Instruction Adherence:\n - Fully Adhered (9-10 points): The response fully complies with all instructions and requirements of the question.\n \
- Partially Adhered (6-8 points): The response meets most of the instructions but has some omissions or misunderstandings.\n\
- Basically Adhered (3-5 points): The response meets some instructions, but the main requirements are not fulfilled.\n\
- Not Adhered (1-2 points): The response does not meet any instructions.\n\
Example: If the question requires three examples and the response provides only one, it falls under "Partially Adhered." \
3. Usefulness:\n - Highly Useful (9-10 points): The response provides comprehensive and accurate information, fully addressing the issue.\n - Useful but Incomplete (6-8 points): \
The response provides some useful information, but lacks details or accuracy.\n - Limited Usefulness (3-5 points): The response offers little useful information, with most content \
being irrelevant or incorrect.\n - Useless or Incorrect (1-2 points): The response is completely \
irrelevant or incorrect.\n Example: If there are factual errors in the response but the overall \
direction is correct, it falls under "Useful but Incomplete." \
4. Relevance:\n - Highly Relevant (9-10 points): The response is highly relevant to the \
question, with information closely aligned with the topic.\n - Generally Relevant (6-8 points): \
The response is generally relevant but includes some unnecessary information.\n - Partially \
Relevant (3-5 points): The response has a lot of content that deviates from the topic.\n - Not \
Relevant (1-2 points): The response is completely irrelevant.\n Example: If the response strays \
from the topic but still provides some relevant information, it falls under "Partially Relevant." \
#### Question Begin ####\n{question}\n \
#### Responses to be Scored #### \
[The Begin of Response 1]\n{response1}\n[The End of Response 1]\n \
[The Begin of Response 2]\n{response2}\n[The End of Response 2]\n \
[The Begin of Response 3]\n{response3}\n[The End of Response 3]\n \
[The Begin of Response 4]\n{response4}\n[The End of Response 4]\n \
[The Begin of Response 5]\n{response5}\n[The End of Response 5]\n 
#### Output Format Requirements #### \
Output with three lines \
Specific Criteria: <Other potential criteria specific to the query and the context, and the \
weights of each criteria>. \
Analysis: <Compare different responses based on given Criteria>. \
Scores: <the overall comprehensive score of all responses in order, separate by comma in the \
boxed, e.g., \\boxed{{x, x, x, x, x}}>. \
"""

NEW_OPTIMIZED_PROMPT = """ You are provided with a problem statement and two candidate responses.  \nYour job is to **evaluate** each response, **compare** them, and assign an overall integer score from\u202f1\u202fto\u202f10 (higher = better) for each one.  \n\n**Instructions**\n\n1. **Read the question** – the text between the markers `#### Question Begin ####` and `#### Question End ####`.  \n   Use the placeholder `{question}` here.\n\n2. **Read the two responses** – the text between the markers `[The Begin of Response 1]` / `[The End of Response 1]` and `[The Begin of Response 2]` / `[The End of Response 2]`.  \n   Use the placeholders `{response1}` and `{response2}` respectively.\n\n3. **Define evaluation criteria** that are relevant to the given question (e.g., correctness, efficiency, code style, explanation clarity, edge‑case handling, etc.).  \n   Assign a weight to each criterion (the weights should sum to 10, but any reasonable scheme is acceptable).  \n\n4. **Analyze step‑by‑step**:  \n   - For each criterion, discuss how each response meets or fails it.  \n   - Compare the two responses directly, citing concrete strengths and weaknesses.  \n\n5. **Score each response** on a scale of 1–10, applying the weights you defined.  \n   The score should reflect the overall quality of the response with respect to all criteria.  \n\n6. **Output** exactly three lines in the format below:\n\n```\nSpecific Criteria: <list each criterion with its weight, e.g., Correctness (4), Efficiency (3), Explanation (2), Edge‑case handling (1)>\nAnalysis: <your detailed, step‑by‑step comparison of the two responses based on the criteria>\nScores: \\boxed{{x, y}}\n```\n\n- Replace `x` with the score for **Response\u202f1** and `y` with the score for **Response\u202f2**.  \n- Use a single space after the comma inside the braces.  \n- Do **not** include any additional text outside the three required lines.\n\n**Remember** to keep the placeholders exactly as `{question}`, `{response1}`, and `{response2}` so that the template can be filled automatically.  \n\nNow evaluate the following inputs:\n\n#### Question Begin ####\n{question}\n#### Question End ####\n\n#### Responses to be Scored ####\n[The Begin of Response 1]\n{response1}\n[The End of Response 1]\n\n[The Begin of Response 2]\n{response2}\n[The End of Response 2]"""

OPTIMIZED_PROMPT_V2 = """You are given a question and two candidate responses. Evaluate them rigorously and assign scores.\n\n#### Question Begin ####\n{question}\n#### Responses Begin ####\n[The Begin of Response 1]\n{response1}\n[The End of Response 1]\n\n[The Begin of Response 2]\n{response2}\n[The End of Response 2]\n\n#### Evaluation Instructions ####\n1. **Define Criteria** – List all criteria that are relevant for judging the quality of the responses (e.g., correctness, efficiency / scalability, completeness, handling of edge‑cases, code quality, adherence to I/O format, clarity of explanation, etc.).  \n   *Assign a weight to each criterion (weights may be expressed as percentages, points out of 10, or any scale that clearly shows their relative importance). The sum of the weights should reflect the total importance (e.g., 100\u202f% or 10 points).*\n\n2. **Step‑by‑Step Analysis** – For each response, examine how well it satisfies every criterion.  \n   *Be explicit: cite where the response meets the criterion, where it falls short, and any ambiguities or errors.*\n\n3. **Scoring** – Using the analysis, give each response an **integer score from 1 to 10** (higher\u202f=\u202fbetter).  \n   *Apply the weights you defined: the final score should reflect the weighted overall quality. Be as strict as possible.*\n\n#### Output Format Requirements ####\nProduce **exactly three lines** in the following order:\n\n**Specific Criteria:** `<list each criterion with its weight, e.g., Correctness\u202f(40\u202f%), Efficiency\u202f(30\u202f%), Clarity\u202f(20\u202f%), Formatting\u202f(10\u202f%)>`\n\n**Analysis:** `<brief comparative analysis of the two responses based on the criteria listed above>`\n\n**Scores:** `\\boxed{{x, x}}`   *(replace\u202fx with the score for Response\u202f1 and Response\u202f2 respectively)*\n\nMake sure the output matches this format precisely, without any extra text or markup."""


OPTIMIZED_PROMPT_V3 = """You are given a question and two candidate responses. Your task is to evaluate each response, assign a strict integer score from 1\u202fto\u202f10 (higher\u202f=\u202fbetter), and present the results in a fixed three‑line format.\n\n**Steps to follow**\n\n1. **Define evaluation criteria** – List the criteria that are most relevant for the given question (e.g., Correctness, Completeness, Efficiency, Robustness, Clarity, Style, etc.).  \n   - Optionally assign a weight to each criterion (the weights can be percentages, fractions, or any relative numbers; they do not need to sum to 100\u202f%).  \n\n2. **Step‑by‑step analysis** – Examine *Response\u202f1* and *Response\u202f2* against every criterion.  \n   - Explain how each response meets or fails the criterion.  \n   - When appropriate, directly compare the two responses.  \n\n3. **Compute composite scores** – Combine the per‑criterion assessments using the stated weights to obtain a single overall score for each response.  \n   - Round the final scores to the nearest integer between 1 and 10.  \n\n4. **Output format** – Provide exactly three lines, in the order shown below, and **no additional text**.\n\n```\nSpecific Criteria: <criterion\u202f1 (weight), criterion\u202f2 (weight), …>\nAnalysis: <concise but thorough comparison of the two responses based on the criteria>\nScores: \\boxed{{x, y}}\n```\n\n- `x` is the score for **Response\u202f1** (`{response1}`)  \n- `y` is the score for **Response\u202f2** (`{response2}`)\n\n**Now evaluate the following items**\n\n#### Question Begin ####\n{question}\n#### Responses to be Scored ####\n[The Begin of Response 1]\n{response1}\n[The End of Response 1]\n[The Begin of Response 2]\n{response2}\n[The End of Response 2]"""

if __name__ == "__main__":
    print("This module is not meant to be run directly. It contains prompts for evaluation tasks.")