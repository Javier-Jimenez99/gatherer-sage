import pandas as pd
from datasets import Dataset
import torch
from transformers import (
    TrainerCallback,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import wandb
import evaluate
import numpy as np
from gatherer_sage.utils import clean_text
import json
import typer
from unsloth import FastLanguageModel

context_instruct_prompt = """<s>[INST] Using the information contained in the context, give a comprehensive and concise answer to the question.
Respond only to the question asked, ensuring that the response is concise and relevant.
Provide the number of the rule when relevant.
If the answer cannot be deduced from the context, do not give an answer.
The questions are related with Magic The Gathering card game.   
Context:
{context}
---
Question: {question}[/INST]{answer}</s>"""

context_instruct_prompt_noanswer = """<s>[INST] Using the information contained in the context, give a comprehensive and concise answer to the question.
Respond only to the question asked, ensuring that the response is concise and relevant.
Provide the number of the rule when relevant.
If the answer cannot be deduced from the context, do not give an answer.
The questions are related with Magic The Gathering card game.   
Context:
{context}
---
Question: {question}[/INST]"""

no_context_instruct_prompt = """<s>[INST] Respond only to the question asked, ensuring that the response is concise and relevant.
Provide the number of the rule when relevant.
If the answer cannot be deduced from the context, do not give an answer.
The questions are related with Magic The Gathering card game.
Question: {question}[/INST]{answer}</s>"""

no_context_instruct_prompt_noanswer = """<s>[INST] Respond only to the question asked, ensuring that the response is concise and relevant.
Provide the number of the rule when relevant.
If the answer cannot be deduced from the context, do not give an answer.
The questions are related with Magic The Gathering card game.
Question: {question}[/INST]"""


def dataset_gen_with_context(data, allow_system_role=True):
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
    use_context: bool = False,
):
    usefull_columns = ["question", "answer"]
    if use_context:
        usefull_columns.append("context")

    df = pd.read_csv(data_path)[usefull_columns].dropna()
    df = df.map(clean_text)

    if num_samples > 0:
        df = df.sample(num_samples, random_state=42)

    # if use_context:
    #     dataset = Dataset.from_generator(
    #         dataset_gen_with_context,
    #         gen_kwargs={"data": df, "allow_system_role": allow_system_role},
    #     )
    # else:
    #     messages = df.apply(
    #         lambda x: {
    #             "messages": [
    #                 {"role": "user", "content": x["question"]},
    #                 {"role": "assistant", "content": f"Answer: {x['answer']}"},
    #             ]
    #         },
    #         axis=1,
    #     )
    #     dataset = Dataset.from_list(messages.tolist())

    def formatting_prompts_func():
        for i, row in df.iterrows():
            if use_context:
                text = context_instruct_prompt.format(
                    context=row["context"],
                    question=row["question"],
                    answer=row["answer"],
                )
            else:
                text = no_context_instruct_prompt.format(
                    question=row["question"], answer=row["answer"]
                )

            yield {"text": text}

    dataset = Dataset.from_generator(formatting_prompts_func)

    dataset = dataset.train_test_split(test_size=0.1, seed=42)
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


def train(
    data_path: str = "data/reddit/reddit_qa_dataset_with_context.csv",
    model_path: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    pretrained_adapter_path: str = None,
    lora_rank: int = 256,
    lora_alpha_scale: float = 0.5,
    learning_rate: float = 2e-4,
    wandb_entity: str = "javier-jimenez99",
    wandb_project: str = "gatherer-sage",
    sweep: str = True,
    eval_ratio: float = 0.1,
    use_context: bool = False,
    batch_size: int = 2,
    run_id: str = None,
    generate_during_eval: bool = False,
):
    base_run_id = run_id
    if base_run_id is not None:
        wandb.init(
            project=wandb_project, entity=wandb_entity, id=base_run_id, resume="allow"
        )
    else:
        wandb.init(project=wandb_project, entity=wandb_entity)

    run_id = wandb.run.id

    if sweep:
        config = wandb.config
        model_path = config.model_path
        lora_rank = config.lora_rank
        lora_alpha_scale = config.lora_alpha_scale
        learning_rate = config.learning_rate

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
        use_context=use_context,
    )

    # BitsAndBytesConfig int-4 config
    if "bnb-4bit" not in model_path.lower():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None

    max_seq_length = 2048

    # Load model and tokenizer
    if pretrained_adapter_path is None:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            lora_alpha=lora_alpha_scale * lora_rank,
            lora_dropout=0.1,
            r=lora_rank,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            use_gradient_checkpointing=True,
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=pretrained_adapter_path,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )

    model.config.use_cache = False
    if (
        "llama" in model_path.lower()
        or "mistral" in model_path.lower()
        or "phi" in model_path.lower()
    ):
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    tokenizer.padding_side = "right"

    model.print_trainable_parameters()

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")

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

        bertscores = bertscore.compute(
            predictions=decoded_preds, references=decoded_labels, lang="en"
        )

        scores["bertscore_f1"] = np.mean(bertscores["f1"])
        scores["bertscore_precision"] = np.mean(bertscores["precision"])
        scores["bertscore_recall"] = np.mean(bertscores["recall"])

        return scores

    gradient_accumulation_steps = 4

    # Training Params
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        logging_steps=1,
        eval_steps=eval_ratio,
        eval_strategy="steps",
        save_steps=eval_ratio,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=2,  # Save two in case there is an old one better
    )

    # Define una función para generar ejemplos de texto y loggearlos en una tabla
    def log_examples(model, tokenizer, step):
        inputs = [
            "Addyson casts Lightning Bolt targeting Nico. As it resolves, can Addyson choose to deal the damage to Nico's Karn, Scion of Urza?",
            "Abdiel casts Desperate Ritual and would like to use the mana it generates to splice another Desperate Ritual onto it. Can they do this?",
            "What happends if a player try to cast a sorcery in their opponent turn?",
            "Allyson casts Thoughtseize targeting Nico. In response, Nico casts their last card, and when Thoughtseize resolves, Nico has zero cards in hand. Does Allyson lose 2 life?",
            "Can a player activate the ability of a creature that has summoning sickness?",
            "Aydin controls a Construct token created by Urza's Saga and 3 other artifacts. If Nico casts Dress Down, what is the size of the Construct?",
            "Aliza casts Blood Moon. Nickolas activates Inkmoth Nexus's ability in response. What will Inkmoth Nexus look like after everything resolves? What will it look like next turn?",
            "Noe controls Urza's Saga on chapter 2. Ariya casts Magus of the Moon. What happens to Urza's Saga?",
            "Alonso controls Goblin Electromancer and is casting Stomp. Does it cost less?",
            "Annabella activates Boseiju, Who Endures targeting Noel's permanent. Can Noel put a Wastes onto the battlefield from their library?",
            "Alyssa controls a creature with deathtouch and trample. If it deals lethal damage to a creature, how much damage can be assigned to the defending player?",
            "John controls a 3/4 creature with deathtouch and he attacks. If the defending player blocks with two 2/4 creatures can John assign 1 damage to each creature and kill both?",
            "In a Two-Headed Giant game, how much life does each team begin the game with?",
            "In a 4-players commander game, how much life does each player begin the game with?",
            "In a commander game how much cards draw each player in their first turn?",
            "Can Ashley attach Cranial Plating at instant speed?",
            "Can Ashley attach an equipment at instant speed?",
            "Can Ashley attach an equipment the turn it enters the game?",
        ]

        table = wandb.Table(columns=["Prompt", "Generated Text"])

        for input_text in inputs:
            inputs_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            outputs = model.generate(**inputs_ids, max_length=max_seq_length)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Añade la fila a la tabla
            table.add_data(input_text, generated_text)

        # Loggea la tabla
        wandb.log({f"example_generation_step_{step}": table})

    # Define el callback de evaluación y logging
    class EvaluationLoggingCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, **kwargs):
            log_examples(model, tokenizer, state.global_step)

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        packing=False,
        callbacks=[EvaluationLoggingCallback] if generate_during_eval else [],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        dataset_text_field="text",
    )

    # Training
    trainer.train(resume_from_checkpoint=base_run_id is not None)
    trainer.evaluate()
    trainer.model.save_pretrained(f"{output_dir}/best_model")


def main(
    sweep_config_path: str = None,
    wadnb_project: str = "gatherer-sage",
    wandb_entity: str = "javier-jimenez99",
    data_path: str = "data/rules_guru/rules_guru_qa_dataset_with_context.csv",  # "data/pretrain_qa_dataset.csv",
    model_path: str = "TrevorJS/mtg-mistral-7b-instruct-sft-merged",  # "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    pretrained_adapter_path: str = "model/mistral-gatherer-sage-v1/full_train_instruct/best_model",  # "model/mistral-gatherer-sage-v1/trevor_full/best_model",
    generate_during_eval: bool = False,
    lora_rank: int = 256,
    lora_alpha_scale: float = 0.5,
    learning_rate: float = 2e-4,
    use_context: bool = False,
    batch_size: int = 16,
    run_id: str = None,  # "y1lel8qp",
):
    if sweep_config_path is not None:
        sweep_configuration = json.load(open(sweep_config_path, "r"))
        sweep_id = wandb.sweep(
            sweep=sweep_configuration,
            project=wadnb_project,
            entity=wandb_entity,
        )
        wandb.agent(sweep_id, function=train, entity=wandb_entity)
    else:
        train(
            data_path=data_path,
            model_path=model_path,
            pretrained_adapter_path=pretrained_adapter_path,
            lora_rank=lora_rank,
            lora_alpha_scale=lora_alpha_scale,
            learning_rate=learning_rate,
            wandb_entity=wandb_entity,
            wandb_project=wadnb_project,
            sweep=False,
            use_context=use_context,
            batch_size=batch_size,
            run_id=run_id,
            generate_during_eval=generate_during_eval,
        )


if __name__ == "__main__":
    typer.run(main)
