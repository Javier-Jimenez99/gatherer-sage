from datasets import load_dataset, concatenate_datasets
import pandas as pd
from gatherer_sage.utils import clean_text


def generate_full_corpus() -> pd.DataFrame:
    """
    It will load, clean and merge these datasets:
    - nelsntk/mtg-data
    - jakeboggs/MTG-Eval
    - RiverTest/Testmtg
    - TrevorJS/mtg-rules-qa
    - Javier-Jimenez99/reddit-mtgrules-qa
    - rules-guru

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns "prompt", "response" and "dataset_name".
    """

    all_df = []

    ds_nelsntk = load_dataset("nelsntk/mtg-data")
    df = pd.DataFrame(concatenate_datasets([v for v in ds_nelsntk.values()]))
    df["dataset_name"] = "nelsntk/mtg-data"
    all_df.append(df)

    ds_jakeboggs = load_dataset("jakeboggs/MTG-Eval")
    df = pd.DataFrame(
        concatenate_datasets([v for v in ds_jakeboggs.values()]).rename_column(
            "instruction", "prompt"
        )
    )
    df["dataset_name"] = "jakeboggs/MTG-Eval"
    all_df.append(df)

    ds_rivertest = load_dataset("RiverTest/Testmtg")
    df = pd.DataFrame(concatenate_datasets([v for v in ds_rivertest.values()]))
    df["dataset_name"] = "RiverTest/Testmtg"
    all_df.append(df)

    ds_trevorjs = load_dataset("TrevorJS/mtg-rules-qa")
    df = pd.DataFrame(
        concatenate_datasets([v for v in ds_trevorjs.values()])
        .rename_column("question", "prompt")
        .rename_column("response_j", "response")
    )
    df["dataset_name"] = "TrevorJS/mtg-rules-qa"
    all_df.append(df)

    df_mtgjudge = load_dataset("Javier-Jimenez99/reddit-mtgrules-qa")
    df = pd.DataFrame(concatenate_datasets([v for v in df_mtgjudge.values()]))
    df["dataset_name"] = "Javier-Jimenez99/reddit-mtgrules-qa"
    all_df.append(df)

    df_rules_guru = pd.read_csv("../data/rules_guru/rules_guru_qa_dataset.csv").rename(
        columns={"question": "prompt", "answer": "response"}
    )
    df_rules_guru["dataset_name"] = "rules_guru"
    all_df.append(df_rules_guru)

    full_ds = pd.concat(all_df, ignore_index=True)[
        ["prompt", "response", "dataset_name"]
    ]
    full_ds = (
        full_ds.dropna().map(clean_text).drop_duplicates(subset=["prompt", "response"])
    )

    return full_ds
