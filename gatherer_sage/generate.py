import typer
import os
from langchain_groq import ChatGroq
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from tqdm import tqdm
from unsloth import FastLanguageModel
from gatherer_sage.train import simple_test_prompt
from transformers import pipeline
import time

tqdm.pandas()

def main(model_path:str, input_path:str, output_path:str,batch_size:int=16):
    dataframe = pd.read_csv(input_path)
    if not "prompt" in dataframe.columns or not "response" in dataframe.columns:
        raise ValueError("Input dataset must have columns 'prompt' and 'response'")
    
    dataframe = dataframe.rename(columns={"prompt": "user_input", "response": "reference"})

    if not os.path.exists(model_path):
        print("Local path to the model not found. Trying with Groq API.")

        print("==== Loading model...")
        chat_groq = ChatGroq(model_name=model_path,max_retries=10)

        prompt_template = """
        You are an expert in Magic: The Gathering. Answer the following question with precise and concise information:
        Question: {user_input}
        Answer:"""

        prompt = PromptTemplate(
            input_variables=["user_input"],
            template=prompt_template,
        )

        chain = prompt | chat_groq | StrOutputParser()

        print("==== Generating answers...")
        for i, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
            model_call = chain.invoke(row["user_input"])
            dataframe.at[i, "response"] = model_call
            time.sleep(1)

        dataframe.to_csv(output_path, index=False)
    else:
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
            use_cache=True,
        )

        prompt_template = tokenizer.apply_chat_template(
            simple_test_prompt, tokenize=False, add_generation_prompt=True
        )

        print("==== Generating answers...")
        def generate(user_input: str):
            return llm_pipeline(prompt_template.format(question=user_input))[0]["generated_text"]

        dataframe["response"] = dataframe["user_input"].progress_apply(generate)

        dataframe.to_csv(
            output_path,
            index=False,
        )

if __name__ == "__main__":
    typer.run(main)