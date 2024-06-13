from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
correctness_rubric = {
    "criteria": "Determine whether the actual output is factually correct based on the expected output.",
    "score1_description": "If the generated answer is not relevant to the user query and reference answer.",
    "score2_description": "If the generated answer is correct according to reference answer but not relevant to user query.",
    "score3_description": "If the generated answer is relevant to the user query and correct according to reference answer but has some mistakes in facts.",
    "score4_description": "If the generated answer is relevant to the user query and has the exact same metrics and correct as the reference answer, but it is not as concise.",
    "score5_description": "If the generated answer is relevant to the user query and fully correct according to the reference answer.",
}

faithfullness_rubric = {
    "criteria": "Does the generated answer accurately represent the information found in the reference material?",
    "score1_description": "If the existing answer is already YES or If the Information is present in the context.",
    "score2_description": "If the existing answer is NO and If the Information is not present in the context.",
}

prometheus_correctness_eval_prompt_template = """###Task Description: An instruction (might include an Input inside it), a query, a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given. 
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general. 
2. After writing a feedback, write a score that is either 1 or 2 or 3 or 4 or 5. You should refer to the score rubric. 
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (1 or 2 or 3 or 4 or 5)" 
4. Please do not generate any other opening, closing, and explanations. 
5. Only evaluate on common things between generated answer and reference answer. Don't evaluate on things which are present in reference answer but not in generated answer.

###The instruction to evaluate: Your task is to evaluate the generated answer and reference answer for the query: {query}

###Generate answer to evaluate: {generated_answer} 

###Reference Answer (Score 5): {reference_answer}

###Score Rubrics: 
Score 1: If the generated answer is not relevant to the user query and reference answer.
Score 2: If the generated answer is correct according to reference answer but not relevant to user query.
Score 3: If the generated answer is relevant to the user query and correct according to reference answer but has some mistakes in facts.
Score 4: If the generated answer is relevant to the user query and has the exact same metrics and correct as the reference answer, but it is not as concise.
Score 5: If the generated answer is relevant to the user query and fully correct according to the reference answer.
    
###Feedback: """

prometheus_faithfulness_refine_prompt_template = """###Task Description: An instruction (might include an Input inside it), a information, a context information, an existing answer, and a score rubric representing a evaluation criteria are given. 
1. You are provided with evaluation task with the help of information, context information and an existing answer.
2. Write a detailed feedback based on evaluation task and the given score rubric, not evaluating in general.
3. After writing a feedback, write a score that is YES or NO. You should refer to the score rubric. 
4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (YES or NO)" 
5. Please do not generate any other opening, closing, and explanations. 

###The instruction to evaluate: If the information is present in the context and also provided with an existing answer.

###Existing answer: {existing_answer} 

###Information: {query_str}

###Context: {context_msg}

###Score Rubrics: 
Score YES: If the existing answer is already YES or If the Information is present in the context.
Score NO: If the existing answer is NO and If the Information is not present in the context.
    
###Feedback: """

prometheus_relevancy_refine_prompt_template = """###Task Description: An instruction (might include an Input inside it), a query with response, context, an existing answer, and a score rubric representing a evaluation criteria are given. 
1. You are provided with evaluation task with the help of a query with response and context and an existing answer.
2. Write a detailed feedback based on evaluation task and the given score rubric, not evaluating in general. 
3. After writing a feedback, write a score that is YES or NO. You should refer to the score rubric. 
4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (YES or NO)" 
5. Please do not generate any other opening, closing, and explanations. 

###The instruction to evaluate: Your task is to evaluate if the response for the query is in line with the context information provided.

###Query and Response: {query_str} 

###Context: {context_str}

###Score Rubrics: 
Score YES: If the existing answer is already YES or If the response for the query is in line with the context information provided.
Score NO: If the existing answer is NO and If the response for the query is in line with the context information provided.

###Feedback: """


def metric_evaluation(
    dataset: pd.DataFrame, judge: PrometheusEval, metric_rubric: dict[str]
):
    score_rubric = SCORE_RUBRIC_TEMPLATE.format(**metric_rubric)

    scores = dataset.progress_apply(
        lambda x: judge.single_absolute_grade(
            instruction=x["question"],
            response=x["generated_answer"],
            rubric=score_rubric,
            reference_answer=x["answer"],
        )[1],
        axis=1,
    ).tolist()
    return scores


def evaluate_generation(dataset: pd.DataFrame):
    model = VLLM(model="prometheus-eval/prometheus-7b-v2.0")
    judge = PrometheusEval(
        model=model,
        absolute_grade_template=ABSOLUTE_PROMPT,
    )

    scores = {"correctness": metric_evaluation(dataset, judge, correctness_rubric)}

    return scores
