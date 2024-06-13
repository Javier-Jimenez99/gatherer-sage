import wandb
import numpy as np
import evaluate
from unsloth import FastLanguageModel
from transformers import pipeline
import pandas as pd
from gatherer_sage.train import (
    context_instruct_prompt_noanswer,
    no_context_instruct_prompt_noanswer,
)

prompt_in_chat_format = [
    {
        "role": "user",
        "content": """Using the information contained in the context,
give a comprehensive and concise answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the rule when relevant.
If the answer cannot be deduced from the context, do not give an answer.
The questions are related with Magic The Gathering card game.
        
Context:
{context}
---
Now here is the question you need to answer.

Question: {question}""",
    },
]

wandb_project = "gatherer-sage"
wandb_entity = "javier-jimenez99"


def get_answer(questions, model, contexts=None, rag=None, batch_size=16):
    if isinstance(questions, str):
        questions = [questions]

    if contexts is None:
        contexts = [contexts] * len(questions)

    all_prompts = []
    for q, c in zip(questions, contexts):
        # If no context is provided, try to retrieve it from the RAG
        # If no context is provided and no RAG is provided, use the no_context_instruct_prompt_noanswer
        if c is None and rag is not None:
            c = rag.retrieve_context(q)

        if c is None:
            prompt = no_context_instruct_prompt_noanswer.format(question=q)
        else:
            prompt = context_instruct_prompt_noanswer.format(context=c, question=q)

        all_prompts.append(prompt)

    gens = model(all_prompts, batch_size=batch_size)

    return all_prompts, [g[0]["generated_text"] for g in gens]


def evaluate_model(model, eval_dataset):
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")

    def compute_metrics(preds, labels):
        scores = rouge.compute(
            predictions=preds,
            references=labels,
            rouge_types=["rouge1", "rouge2", "rougeL"],
            use_aggregator=True,
            use_stemmer=True,
        )

        scores["bleu"] = bleu.compute(predictions=preds, references=labels)["bleu"]

        bertscores = bertscore.compute(predictions=preds, references=labels, lang="en")

        scores["bertscore_f1"] = np.mean(bertscores["f1"])
        scores["bertscore_precision"] = np.mean(bertscores["precision"])
        scores["bertscore_recall"] = np.mean(bertscores["recall"])

        return scores

    full_questions, generated_answers = get_answer(
        eval_dataset["question"],
        model,
        contexts=eval_dataset["context"] if "context" in eval_dataset.columns else None,
    )

    df_outputs = (
        eval_dataset[["question", "answer"]]
        .copy()
        .rename(columns={"answer": "real_answer"})
    )
    df_outputs["full_question"] = full_questions
    df_outputs["generated_answer"] = generated_answers

    scores = compute_metrics(
        df_outputs["generated_answer"].to_list(), df_outputs["real_answer"].to_list()
    )

    return df_outputs, scores


def get_only_scores(df_scores):
    df = df_scores.copy()
    df = df.drop(
        columns=["question", "full_question", "real_answer", "generated_answer"]
    )

    return df.mean()


def main(
    model_path: str = "model/mistral-gatherer-sage-v1/rules_guru_v5.2/best_model",
    dataset_path: pd.DataFrame = "data/rules_guru/rules_guru_qa_dataset_with_context.csv",
):
    wandb.init(project=wandb_project, entity=wandb_entity)

    test_dataset = pd.read_csv(dataset_path).sample(frac=0.1, random_state=42)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    # model.config.use_cache = False
    tokenizer.pad_token = tokenizer.eos_token

    FastLanguageModel.for_inference(model)

    config = wandb.config

    llm_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=config.temperature,
        repetition_penalty=config.repetition_penalty,
        top_k=config.top_k,
        top_p=config.top_p,
        return_full_text=False,
        max_new_tokens=500,
    )

    df_outputs, scores = evaluate_model(llm_pipeline, tokenizer, test_dataset)

    scores["Scores Table"] = wandb.Table(dataframe=df_outputs)

    wandb.log(scores)


if __name__ == "__main__":
    sweep_configuration = {
        "method": "bayes",
        "name": "generation-finetuning-sweep",
        "metric": {"goal": "maximize", "name": "bertscore_f1"},
        "parameters": {
            "top_k": {"min": 10, "max": 100, "distribution": "int_uniform"},
            "top_p": {"min": 0.5, "max": 1.0, "distribution": "uniform"},
            "temperature": {"min": 0.3, "max": 2, "distribution": "uniform"},
            "repetition_penalty": {"min": 1.0, "max": 2.0, "distribution": "uniform"},
        },
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project=wandb_project,
        entity=wandb_entity,
    )
    wandb.agent(sweep_id, function=main, entity=wandb_entity)
