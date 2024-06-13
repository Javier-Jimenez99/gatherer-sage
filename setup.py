import setuptools

setuptools.setup(
    name="gatherer_sage",
    use_scm_version=True,
    author="Javier Jimenez",
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    setup_requires=["setuptools_scm"],
    install_requires=[
        "pandas",
        "langchain",
        "transformers",
        "PyPDF2",
        "matplotlib",
        "torch",
        "huggingface_hub",
        "langchain_community",
        "sentence-transformers",
        "faiss-gpu",
        "bitsandbytes",
        "accelerate",
        "ragatouille",
        "evaluate",
        "rouge_score",
        "wandb",
        "einops",  # phi
        "pytest",  # phi
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "xformers",
        "bert-score",
        "prometheus-eval",
        "typer",
        "vllm",
        "peft",
        "trl",
        "pandas",
    ],
)
