import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
import wandb
import evaluate
import numpy as np
from gatherer_sage.utils import clean_text


def dataset_gen(data, allow_system_role=True):
    for idx, row in data.iterrows():
        if allow_system_role:
            prompt = [
                {
                    "role": "system",
                    "content": """Using the information contained in the context,
give a comprehensive and concise answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the rule when relevant.
If the answer cannot be deduced from the context, do not give an answer.
The questions are related with Magic The Gathering card game.""",
                },
                {
                    "role": "user",
                    "content": f"""Context:
{row['context']}
---
Now here is the question you need to answer.

Question: {row['question']}""",
                },
                {"role": "assistant", "content": f"Answer: {row['answer']}"},
            ]
        else:
            prompt = [
                {
                    "role": "user",
                    "content": f"""Using the information contained in the context,
give a comprehensive and concise answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the rule when relevant.
If the answer cannot be deduced from the context, do not give an answer.
The questions are related with Magic The Gathering card game.

Context:
{row['context']}
---
Now here is the question you need to answer.

Question: {row['question']}""",
                },
                {"role": "assistant", "content": f"Answer: {row['answer']}"},
            ]

        yield {"messages": prompt}


def create_datasets(
    data_path: str = "data/reddit/reddit_qa_dataset_with_context.csv",
    num_samples: int = -1,
    allow_system_role: bool = True,
):
    reddit_df = pd.read_csv(data_path).dropna()
    reddit_df["question"] = reddit_df["question"].apply(clean_text)
    reddit_df["answer"] = reddit_df["answer"].apply(clean_text)
    reddit_df["context"] = reddit_df["context"].apply(clean_text)

    if num_samples > 0:
        reddit_df = reddit_df.sample(num_samples, random_state=42)

    dataset = Dataset.from_generator(
        dataset_gen,
        gen_kwargs={"data": reddit_df, "allow_system_role": allow_system_role},
    )
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    return train_dataset, test_dataset


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def main(
    data_path: str = "data/reddit/reddit_qa_dataset_with_context.csv",
    model_path: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    lora_rank: int = 256,
    lora_alpha_scale: float = 0.5,
    learning_rate: float = 2e-4,
):
    wandb.init(project="gatherer-sage", entity="javier-jimenez99")
    run_id = wandb.run.id
    config = wandb.config
    model_path = config.model_path
    # lora_rank = config.lora_rank
    lora_alpha_scale = config.lora_alpha_scale
    # learning_rate = config.learning_rate

    if "llama" in model_path.lower():
        model_base_name = "llama"
    elif "phi" in model_path.lower():
        model_base_name = "phi"
    elif "gemma" in model_path.lower():
        model_base_name = "gemma"
    elif "mistral" in model_path.lower():
        model_base_name = "mistral"
    else:
        model_base_name = "model"

    output_dir = f"./model/{model_base_name}-gatherer-sage-v1/{run_id}"

    train_dataset, test_dataset = create_datasets(
        data_path,
        num_samples=-1,
        allow_system_role=model_base_name in ["llama", "phi"],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if (
        "llama" in model_path.lower()
        or "mistral" in model_path.lower()
        or "phi" in model_path.lower()
    ):
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        lora_alpha=lora_alpha_scale * lora_rank,
        lora_dropout=0.05,
        r=lora_rank,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # FIX: This is a temporary fix because there is no generation algorithm
        # So we just return the argmax of the logits
        # There is a trainer parameter called: `preprocess_logits_for_metrics`
        # preds = preds.argmax(-1)

        # decode preds and labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        scores = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            rouge_types=["rouge1", "rouge2", "rougeL"],
            use_aggregator=True,
            use_stemmer=True,
        )

        scores["bleu"] = bleu.compute(
            predictions=decoded_preds, references=decoded_labels
        )["bleu"]

        return scores

    # Training Params
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=1,
        eval_steps=100,
        eval_strategy="steps",
        save_steps=100,
        learning_rate=learning_rate,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=1,  # Save only the most recent checkpoint
    )

    # Trainer
    max_seq_length = 3072  # max sequence length for model and packing of the dataset

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        # packing=True,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        },
    )

    # Training
    trainer.train()
    trainer.model.save_pretrained(f"{output_dir}/best_model")
    trainer.evaluate()


sweep_configuration = {
    "method": "grid",
    "name": "lora-exploration-llama",
    "metric": {"goal": "maximize", "name": "eval/rougeL"},
    "parameters": {
        "lora_alpha_scale": {"values": [0.5]},
        "model_path": {
            "values": [
                # "microsoft/Phi-3-small-8k-instruct",
                # "google/gemma-1.1-7b-it",
                "mistralai/Mistral-7B-Instruct-v0.3",
                # "meta-llama/Meta-Llama-3-8B-Instruct",
            ]
        },
    },
}

sweep_id = wandb.sweep(
    sweep=sweep_configuration, project="gatherer-sage", entity="javier-jimenez99"
)
wandb.agent(sweep_id, function=main, entity="javier-jimenez99")
