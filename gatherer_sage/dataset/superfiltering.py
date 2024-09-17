import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from apricot.functions.facilityLocation import FacilityLocationSelection
import numpy as np


tqdm.pandas()


def get_perplexity_and_embedding_whole_text(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    text: str,
    max_length: int,
    device: str = "cuda",
) -> float:
    """
    Computes the perplexity of the text passed.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        The tokenizer used to encode the text.
    model : AutoModelForCausalLM
        The model used to get the perplexity.
    text : str
        The text to get the perplexity of.
    max_length : int
        The maximum length of the text.
    device : str, optional
        The device to run the model on, by default "cuda".

    Returns
    -------
    float
        The perplexity of the text.
    """
    try:
        input_ids = tokenizer.encode(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids.contiguous())
        loss = outputs.loss
        perplexity = torch.exp(loss)

        return perplexity.to("cpu").item()

    except Exception as e:
        print(e)
        return np.nan, np.nan


def get_perplexity_and_embedding_part_text(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    text: str,
    target_span: str,
    max_length: int,
    device: str = "cuda",
) -> float:
    """
    Computes the perplexity of the end of the text taking into account the begining.
    The text is divided by the target_span.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        The tokenizer used to encode the text.
    model : AutoModelForCausalLM
        The model used to get the perplexity.
    text : str
        The text to get the perplexity of.
    target_span : str
        The target span to split the text.
    max_length : int
        The maximum length of the text.
    device : str, optional
        The device to run the model on, by default "cuda".

    Returns
    -------
    float
        The perplexity of the text.
    """

    try:
        input_ids = tokenizer.encode(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(device)

        start_index = text.rfind(target_span)
        start_token = len(tokenizer.encode(text[:start_index]))

        labels = input_ids.clone()
        labels[0, :start_token] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=labels)

        loss = outputs.loss
        perplexity = torch.exp(loss)

        return perplexity.to("cpu").item()

    except Exception as e:
        print(e)
        return np.nan, np.nan


def compute_ifd(
    row: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_length: int,
) -> tuple[float, float, float]:
    """
    Computes the Instruction-Following Difficultyy (IFD) of the text.

    Parameters
    ----------
    row : pd.DataFrame
        Row of a DataFrame containing the text and the response.
    tokenizer : AutoTokenizer
        The tokenizer used to encode the text.
    model : AutoModelForCausalLM
        The model used to get the perplexity.
    max_length : int
        The maximum length of the text.

    Returns
    -------
    tuple[float, float, float]
        The IFD, the perplexity of the response and the perplexity of the response based on the question.
    """

    ppl_xy = get_perplexity_and_embedding_part_text(
        tokenizer, model, row["full_text"], row["response"], max_length
    )

    ppl_y = get_perplexity_and_embedding_whole_text(
        tokenizer, model, row["response"], max_length
    )

    ifd = ppl_xy / ppl_y

    return ifd, ppl_y, ppl_xy


def superfiltering(
    corpus_path: str = "./data/huge_corpus/train.csv", psize: float = 0.5
) -> pd.DataFrame:
    """
    Superfiltering of the corpus.

    Parameters
    ----------
    corpus_path : str, optional
        The path to the corpus, by default "./data/huge_corpus/train.csv"
    psize : float, optional
        The size of the filtered corpus, by default 0.5
        If psize <= 1, it is considered as a percentage of the original corpus.
        If psize > 1, it is considered as the number of samples to keep.

    Returns
    -------
    pd.DataFrame
        The filtered corpus.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        output_hidden_states=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    max_length = min(1024, tokenizer.model_max_length)
    print("Max length:", max_length)

    # Load the data
    df = pd.read_csv(corpus_path).reset_index(drop=True).dropna()
    df["full_text"] = df["prompt"] + " " + df["response"]

    df[["ifd", "perplexity_y", "perplexity_xy"]] = df.progress_apply(
        compute_ifd,
        axis=1,
        result_type="expand",
        args=(
            tokenizer,
            model,
            max_length,
        ),
    )

    df = df.dropna()
    df = df[df["ifd"] < 1].sort_values("ifd")
    if psize <= 1:
        df = df.head(int(len(df) * psize))
    else:
        df = df.head(psize)

    return df.reset_index(drop=True)


def diversity(
    df: pd.DataFrame,
    embeddings_model: str = "mixedbread-ai/mxbai-embed-large-v1",
    psize: float = 0.1,
) -> pd.DataFrame:
    """
    Diversity filtering of the corpus using Facility Location.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter.
    embeddings_model : str, optional
        The model to use to get the embeddings, by default "mixedbread-ai/mxbai-embed-large-v1"
    psize : float, optional
        The size of the filtered corpus, by default 0.1
        If psize <= 1, it is considered as a percentage of the original corpus.
        If psize > 1, it is considered as the number of samples to keep.

    Returns
    -------
    pd.DataFrame
        The filtered corpus.
    """

    df = df.copy()

    model = SentenceTransformer(embeddings_model)

    embeddings = model.encode(
        df["full_text"].tolist(),
        batch_size=512,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    df["embeddings"] = [embeddings[i] for i in range(embeddings.shape[0])]
    embeddings = np.stack(df["embeddings"].to_list())

    if psize <= 1:
        n_samples = int(len(df) * psize)
    else:
        n_samples = psize

    selector = FacilityLocationSelection(n_samples=n_samples, verbose=True)
    X, y = selector.fit_transform(embeddings, y=df.index.to_numpy())
    return df.iloc[y]
