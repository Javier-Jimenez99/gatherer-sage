import wandb
from gatherer_sage.rag import RAG
from unsloth import FastLanguageModel
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from gatherer_sage.llm_judge import evaluate_generation
from gatherer_sage.utils import gpu_cleaning
import torch
import typer

tqdm.pandas()

wandb_project = "gatherer-sage"
wandb_entity = "javier-jimenez99"

context_prompt = [
    {
        "role": "user",
        "content": """Using the information contained in the context, give a comprehensive and concise answer to the question.
Respond only to the question asked, ensuring that the response is concise and relevant.
Provide the number of the rule when relevant.
If the answer cannot be deduced from the context, do not give an answer.
The questions are related with Magic The Gathering card game.   
Context:
{context}
---
Question: {question}
""",
    }
]


@torch.no_grad
def main(
    model_path: str = "model/carbonbeagle-11b-truthy-gatherer-sage-v2/full_train/last_model",
    data_path: str = "data/huge_corpus/test.csv",
    batch_size: int = 16,
    rerank_model: str = "colbert-ir/colbertv2.0",
    embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1",
    docs_retrieved: int = 10,
    chunk_size: int = 128,
    data_frac: float = 1,
):
    wandb.init(project=wandb_project, entity=wandb_entity)
    config = wandb.config

    model_path = config.get("model_path", model_path)
    data_path = config.get("data_path", data_path)
    batch_size = config.get("batch_size", batch_size)
    rerank_model = config.get("rerank_model", rerank_model)
    embedding_model = config.get("embedding_model", embedding_model)
    docs_retrieved = config.get("docs_retrieved", docs_retrieved)
    chunk_size = config.get("chunk_size", chunk_size)

    print("==== Creating RAG...")
    rag = RAG(
        embedding_model_path=embedding_model,
        reranker_model_path=rerank_model,
        chunk_size=chunk_size,
    )

    print("==== Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    tokenizer.pad_token = tokenizer.eos_token

    FastLanguageModel.for_inference(model)
    model.eval()

    llm_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.31,
        repetition_penalty=1.02,
        top_k=75,
        top_p=0.59,
        return_full_text=False,
        max_new_tokens=500,
    )

    prompt_template = tokenizer.apply_chat_template(
        context_prompt, tokenize=False, add_generation_prompt=True
    )

    print("==== Retrieving context...")
    df = (
        pd.read_csv(data_path)
        .sample(frac=data_frac, random_state=42)
        .rename(columns={"prompt": "question", "response": "answer"})
    )
    df["context"] = df["question"].progress_apply(
        lambda x: rag.retrieve_context(question=x, num_docs_final=docs_retrieved)
    )

    df["input_text"] = df.apply(
        lambda x: prompt_template.format(context=x["context"], question=x["question"]),
        axis=1,
    )

    print("==== Generating answers...")
    df["generated_answer"] = [
        gen[0]["generated_text"]
        for gen in llm_pipeline(df["input_text"].tolist(), batch_size=batch_size)
    ]

    # Cleaning up
    del model
    del tokenizer
    del llm_pipeline
    del rag
    gpu_cleaning()

    print("==== Evaluating answers...")
    scores = evaluate_generation(dataset=df)
    df["correctness"] = scores["correctness"]

    print("==== Saving results...")
    wandb.log({"correctness": df["correctness"].mean()})
    wandb.log({"Scores Table": wandb.Table(dataframe=df)})
    gpu_cleaning()


def get_best_parameter(sweep_id, parameter_name):
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    best_run = sweep.best_run()
    return best_run.config[parameter_name]


def run_sweep(parameter, sweep_config, sweep_id=None):
    sweep_id = (
        wandb.sweep(sweep_config, project=wandb_project, entity=wandb_entity)
        if sweep_id is None
        else sweep_id
    )
    wandb.agent(sweep_id, function=main)

    return get_best_parameter(sweep_id, parameter)


if __name__ == "__main__":
    typer.run(main)

    # sweep_configuration = {
    #     "method": "grid",
    #     "name": "generation-finetuning-sweep",
    #     "metric": {"goal": "maximize", "name": "correctness"},
    #     "parameters": {},
    # }

    # base_parameters = {
    #     "model_path": {
    #         "values": [
    #             "model/mistral-gatherer-sage-v1/rules_guru_context_full/best_model"
    #         ]
    #     },
    #     "docs_retrieved": {"values": [5]},
    #     "chunk_size": {"values": [128]},
    #     "embedding_model": {"values": ["thenlper/gte-large"]},
    #     "rerank_model": {"values": [None]},
    # }

    # stages = {
    #     "model_path": {
    #         "values": [
    #             "jakeboggs/MTG-Llama",
    #             "TrevorJS/mtg-mistral-7b-instruct-sft-merged",
    #             "model/mistral-gatherer-sage-v1/full_train_instruct/best_model",
    #             "model/mistral-gatherer-sage-v1/rules_guru_context_full/best_model",
    #         ]
    #     },
    #     "docs_retrieved": {"values": [5, 10, 15]},
    #     "chunk_size": {"values": [512, 256, 128]},
    #     "embedding_model": {
    #         "values": [
    #             "mixedbread-ai/mxbai-embed-large-v1",
    #             "thenlper/gte-large",
    #             "Alibaba-NLP/gte-large-en-v1.5",
    #         ]
    #     },
    #     "rerank_model": {
    #         "values": ["colbert-ir/colbertv2.0", None, "jinaai/jina-colbert-v1-en"]
    #     },
    # }

    # for param, values in stages.items():
    #     sweep_configuration["parameters"] = base_parameters
    #     sweep_configuration["parameters"][param] = values

    #     sweep_configuration["name"] = f"{param}-sweep"

    #     best_value = run_sweep(param, sweep_configuration)
    #     print(f"Best parameter for {param}: {best_value}")

    #     base_parameters[param] = {"values": [best_value]}
