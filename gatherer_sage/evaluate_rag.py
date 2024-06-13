import wandb
from gatherer_sage.rag import RAG
from unsloth import FastLanguageModel
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from gatherer_sage.llm_judge import evaluate_generation
import torch
import gc

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
    model_path: str = "model/mistral-gatherer-sage-v1/rules_guru_no_context_full/best_model",
    data_path: str = "data/rules_guru/rules_guru_qa_dataset.csv",
    batch_size: int = 16,
):
    wandb.init(project=wandb_project, entity=wandb_entity)
    config = wandb.config

    print("==== Creating RAG...")
    rag = RAG(
        embedding_model_path=config.embedding_model,
        reranker_model_path=config.rerank_model,
        chunk_size=config.chunk_size,
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
    df = pd.read_csv(data_path).sample(frac=0.1, random_state=42)
    df["context"] = df["question"].progress_apply(
        lambda x: rag.retrieve_context(question=x, num_docs_final=config.docs_retrieved)
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
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

    print("==== Evaluating answers...")
    scores = evaluate_generation(dataset=df)
    df["correctness"] = scores["correctness"]

    print("==== Saving results...")
    wandb.log({"correctness": df["correctness"].mean()})
    wandb.log({"Scores Table": wandb.Table(dataframe=df)})


if __name__ == "__main__":
    sweep_configuration = {
        "method": "bayes",
        "name": "generation-finetuning-sweep",
        "metric": {"goal": "maximize", "name": "correctness"},
        "parameters": {
            "embedding_model": {
                "values": [
                    "thenlper/gte-large",
                    "Alibaba-NLP/gte-large-en-v1.5",
                    "mixedbread-ai/mxbai-embed-large-v1",
                ]
            },
            "chunk_size": {"values": [128, 256, 512, 1024]},
            "rerank_model": {
                "values": [None, "colbert-ir/colbertv2.0", "jinaai/jina-colbert-v1-en"]
            },
            "docs_retrieved": {"values": [5, 7, 10]},
        },
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project=wandb_project,
        entity=wandb_entity,
    )
    wandb.agent(sweep_id, function=main, entity=wandb_entity)
