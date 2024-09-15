import pandas as pd
import json
import re
from gatherer_sage.utils import clean_text
from pathlib import Path

from datasets import Dataset
import typer


def read_reddit_file(path: str) -> pd.DataFrame:
    """
    Read a JSON file from reddit data and return a cleaned DataFrame.

    Parameters
    ----------
    path : str
        Path to the Reddit file.

    Returns
    -------
    pd.DataFrame
        DataFrame with the Reddit data.
    """

    with open(path) as f:
        lines = [json.loads(l) for l in f.readlines()]
    df = pd.DataFrame.from_dict(lines)
    df["date"] = pd.to_datetime(
        df["created_utc"].astype(int) * 1e9, utc=True
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.sort_values("date")

    return df


def generate_reddit_dataframe(input_folder: str = "./data/reddit/") -> pd.DataFrame:
    """
    Generate a Q&A dataframe from Reddit data.
    It will clean the text and pick questions and answers pairs with at least one upvote.
    If there are multiple answers, it will pick the one with the most upvotes.

    Parameters
    ----------
    input_folder : str, optional
        Path to the raw Reddit data folder, by default "./data/reddit/"

    Returns
    -------
    pd.DataFrame
        Q&A DataFrame with the Reddit data.
    """

    input_folder = Path(input_folder)
    submissions_df = read_reddit_file(input_folder / "mtgrules_submissions.json")
    submissions_df["full_text"] = (
        submissions_df["title"] + "\n" + submissions_df["selftext"]
    )

    comments_df = read_reddit_file(input_folder / "mtgrules_comments.json")
    comments_df = comments_df[comments_df["author"] != "MTGCardFetcher"]
    comments_df["link_id"] = comments_df["link_id"].str.replace("t3_", "")
    comments_df = comments_df[comments_df["parent_id"].str.contains("t3_")]
    comments_df

    max_upvotes_comments = comments_df.groupby("link_id")["score"].idxmax()
    selected_comments = comments_df.loc[max_upvotes_comments]

    all_df = submissions_df.merge(
        selected_comments,
        left_on="id",
        right_on="link_id",
        suffixes=("_submission", "_comment"),
    )

    all_df = all_df[all_df["score_comment"] > 0]

    df_qa = all_df[["full_text", "body", "score_comment"]]
    df_qa = df_qa.rename(
        columns={"full_text": "question", "body": "answer", "score_comment": "score"}
    )
    df_qa = df_qa.dropna()

    df_qa["question"] = df_qa["question"].apply(clean_text)
    df_qa["answer"] = df_qa["answer"].apply(clean_text)

    df_qa = df_qa.dropna()
    df_qa["question"] = df_qa["question"].apply(
        lambda x: None if re.search(r"\[deleted\]", x) else x
    )
    df_qa["answer"] = df_qa["answer"].apply(
        lambda x: None if re.search(r"\[deleted\]", x) else x
    )
    df_qa = df_qa.dropna().reset_index(drop=True)

    df_qa = df_qa.rename(columns={"question": "prompt", "answer": "response"})

    return df_qa


def main(
    input_folder: str = "./data/reddit/",
    output_path: str = "./data/reddit/reddit_qa_dataset.csv",
    huggingface_hub_path: str = None,
):
    """
    Generate a Q&A dataset from Reddit data.

    Parameters
    ----------
    input_folder : str, optional
        Path to the raw Reddit data folder, by default "./data/reddit/"
    output_path : str, optional
        Path to save the reddit dataset on a CSV file, by default "./data/reddit/reddit_qa_dataset.csv"
    huggingface_hub_path : str, optional
        Path to upload the dataset in the Hugging Face Hub, by default None.
        If None, it won't upload it.
    """

    df_qa = generate_reddit_dataframe(input_folder)
    df_qa.to_csv(output_path, index=False)

    if huggingface_hub_path is not None:
        ds = Dataset.from_pandas(df_qa)
        ds.push_to_hub(huggingface_hub_path)


if __name__ == "__main__":
    typer.run(main)
