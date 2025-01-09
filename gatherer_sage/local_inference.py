import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from gatherer_sage.utils import gpu_cleaning
from langchain_huggingface import HuggingFacePipeline


def optimize_memory(model):
    """
    Implements memory optimization techniques for better model performance.
    """
    gpu_cleaning()

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Use memory efficient attention
    model.config.use_cache = False


def setup_model(model_path):
    """
    Loads Llama 3.3 70B with 4-bit quantization and proper device mapping.
    Returns initialized model and tokenizer.
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",  # Automatically handle multi-GPU
        torch_dtype=torch.bfloat16,
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
        },
    )

    optimize_memory(model)

    return model, tokenizer


def langchain_local_model_from_huggingface(model_path):
    model, tokenizer = setup_model(model_path)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        top_k=50,
        temperature=0.1,
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm
