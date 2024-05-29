from PyPDF2 import PdfReader
import re
import pandas as pd
import json
import numpy as np

def clean_text(text):
    # Protect card names in double brackets
    card_names = re.findall(r'\[\[.*?\]\]', text)
    card_dict = {f'<<{i}>>': card_names[i] for i in range(len(card_names))}
    for key, value in card_dict.items():
        text = text.replace(value, key)
    
    # Remove Markdown URLs
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    
    # Remove standalone URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove Markdown syntax (bold, italic)
    text = re.sub(r'\*{1,2}|_{1,2}', '', text)
    
    # Restore card names
    for key, value in card_dict.items():
        text = text.replace(key, value)

    return text

def get_pdf_content(documents):
    raw_texts = []

    for document in documents:
        raw_text = ""
        pdf_reader = PdfReader(document)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
        
        raw_texts.append(clean_text(raw_text))

    return raw_texts

def load_cards_df(data_path:str="data/AtomicCards.json"):
    all_cards_json = json.load(open(data_path, encoding="utf8"))["data"]

    all_cards = []
    for name, value in all_cards_json.items():
        for i, card in enumerate(value):
            new_name = name
            if i != 0:
                new_name = f"{name}_{i}"

            parsed_card = card
            parsed_card["name"] = new_name

            if "faceName" not in card.keys():
                parsed_card["faceName"] = new_name

            all_cards.append(parsed_card)

    df = pd.DataFrame(all_cards)
        
    renames = {col: col.strip() for col in df.columns}
    df = df.rename(columns=renames)

    return df

def parse_mana_cost(mana_cost):
    mana_parsed = ""
    for v in mana_cost[:-1]:
        if v == "W":
            mana_parsed += "white"
        elif v == "U":
            mana_parsed += "blue"
        elif v == "B":
            mana_parsed += "black"
        elif v == "R":
            mana_parsed += "red"
        elif v == "G":
            mana_parsed += "green"
        elif v == "P":
            mana_parsed += "life"
        elif v == "T":
            mana_parsed += "tap"
        elif v == "{":
            continue
        elif v == "}":
            mana_parsed += ", "
        elif v.isdigit() or v == "X":
            mana_parsed += f"{v} colorless"
        else:
            mana_parsed += " "

    return mana_parsed

def card_texts(cards_df):
    groups = cards_df.groupby("name")
    all_texts = []
    for name, group in groups:
        for index, variation in group.iterrows():
            name = f'Name: {variation["name"]}'

            mana_cost = (
                parse_mana_cost(variation["manaCost"])
                if not pd.isna(variation["manaCost"])
                else ""
            )
            mana_cost = f'Mana Cost: {mana_cost}'

            card_type = f'Type: {variation["type"]}'

            text = variation["text"] if not pd.isna(variation["text"]) else ""
            mana_in_text = re.findall(r"\{.*\}", text)
            for mana in mana_in_text:
                text = text.replace(mana, parse_mana_cost(mana))
            text = f'Text: {text}'

            if not variation["power"] is np.nan or not variation["toughness"] is np.nan:
                stats = f"Stats: {variation['power']} power, {variation['toughness']} toughness"
            else:
                stats = ""

            rulings = variation["rulings"]
            if not rulings is np.nan:
                rules = f"Rules:\n"
                for i,rule in enumerate(rulings):
                    rules += f"{i+1}. {rule['text']}\n"
            else:
                rules = ""

            input_text = ("\n".join([name, mana_cost, card_type, text, stats,rules])).strip()

            all_texts.append(clean_text(input_text))

    return all_texts