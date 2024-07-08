import wandb
from gatherer_sage.rag import RAG
from unsloth import FastLanguageModel
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from gatherer_sage.train import simple_test_prompt
from gatherer_sage.llm_judge import evaluate_generation
from gatherer_sage.utils import gpu_cleaning
import torch
import evaluate
import json
import typer

tqdm.pandas()

wandb_project = "gatherer-sage"
wandb_entity = "javier-jimenez99"

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")


def compute_metrics(preds, labels):
    scores = rouge.compute(
        predictions=preds,
        references=labels,
        rouge_types=["rouge1", "rouge2", "rougeL"],
        use_aggregator=True,
        use_stemmer=True,
    )

    scores["bleu"] = bleu.compute(predictions=preds, references=labels)["bleu"]

    return scores


@torch.no_grad
def main(
    model_path: str = "model/carbonbeagle-11b-truthy-gatherer-sage-v2/subset_20_train/last_model",
    output_path: str = "model/carbonbeagle-11b-truthy-gatherer-sage-v2/subset_20_train/",
    data_path: str = "data/huge_corpus/test.csv",
    batch_size: int = 16,
):
    wandb.init(project=wandb_project, entity=wandb_entity)
    config = wandb.config

    model_path = config.get("model_path", model_path)

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
        simple_test_prompt, tokenize=False, add_generation_prompt=True
    )

    df = pd.read_csv(data_path).rename(
        columns={"prompt": "question", "response": "answer"}
    )

    df["input_text"] = df.apply(
        lambda x: prompt_template.format(question=x["question"]),
        axis=1,
    )

    print("==== Generating answers...")
    df["generated_answer"] = [
        gen[0]["generated_text"]
        for gen in llm_pipeline(df["input_text"].tolist(), batch_size=batch_size)
    ]

    df.to_csv(
        output_path + "predictions.csv",
        index=False,
    )

    print("==== Evaluating answers...")
    scores = compute_metrics(df["generated_answer"].to_list(), df["answer"].to_list())

    with open(output_path + "scores.json", "w") as f:
        json.dump(scores, f)

    # Cleaning up
    del model
    del tokenizer
    del llm_pipeline
    gpu_cleaning()

    print("==== Evaluating answers...")
    scores = evaluate_generation(dataset=df)
    df["correctness"] = scores["correctness"]

    print("==== Saving results...")
    scores["correctness"] = df["correctness"].mean()
    wandb.log(scores)
    wandb.log({"Scores Table": wandb.Table(dataframe=df)})
    gpu_cleaning()

    df.to_csv(
        output_path + "predictions.csv",
        index=False,
    )

    with open(output_path + "scores.json", "w") as f:
        json.dump(scores, f)


if __name__ == "__main__":
    typer.run(main)
