import typer
from ragas.evaluation import evaluate
from ragas.metrics import (
    RougeScore,
    SemanticSimilarity,
    FactualCorrectness,
    AnswerRelevancy,
)
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset, RunConfig
import pandas as pd
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from gatherer_sage.local_inference import langchain_local_model_from_huggingface


AVAILABLE_METRICS = {
    "rouge": RougeScore(),
    "semantic_similarity": SemanticSimilarity(),
    "factual_correctness": FactualCorrectness(),
    "answer_relevancy": AnswerRelevancy(),
}


def main(
    input_path: str,
    output_path: str,
    critic_model: str = "meta-llama/Llama-3.3-70B-Instruct",
    embedding_model: str = "jinaai/jina-embeddings-v3",
    metrics: str = "rouge,semantic_similarity,factual_correctness",
    top: int = -1,
):
    metrics = [AVAILABLE_METRICS[m] for m in metrics.split(",")]

    print("==== Loading input dataset...")
    df = (
        pd.read_csv(input_path)
        .sort_values(by="user_input")
        .sample(n=top, random_state=42)
    )
    ds = EvaluationDataset.from_pandas(df)

    print("==== Loading critic model...")
    # chat = ChatGroq(model_name=critic_model)
    # llm = LangchainLLMWrapper(chat)
    llm = langchain_local_model_from_huggingface(critic_model)

    print("==== Loading embedding model...")
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cuda", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": False},
    )

    print("==== Evaluating...")
    run_config = RunConfig(max_workers=1)
    scores = evaluate(
        dataset=ds,
        metrics=metrics,
        llm=llm,
        embeddings=embedding_model,
        run_config=run_config,
    )

    print("==== Saving results...")
    scores.to_pandas().to_csv(output_path, index=False)


if __name__ == "__main__":
    typer.run(main)
