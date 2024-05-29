from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
import os
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from ragatouille import RAGPretrainedModel

from transformers import AutoTokenizer
from gatherer_sage.utils import get_pdf_content, load_cards_df, card_texts


class RAG:
    def __init__(
        self,
        rules_folder: str = "data/rules/",
        cards_path="data/AtomicCards.json",
        embedding_model_path="thenlper/gte-large",
        chunk_size: int = 512,
        reranker_model_path="colbert-ir/colbertv2.0",
    ):
        self.chunk_size = chunk_size
        self.embedding_model_path = embedding_model_path
        self.reranker_model_path = reranker_model_path

        doc_paths = [rules_folder + f for f in os.listdir(rules_folder)]
        self.langchaing_docs = [
            LangchainDocument(doc) for doc in get_pdf_content(doc_paths)
        ]

        cards_df = load_cards_df(cards_path)

        all_cards_texts = card_texts(cards_df)

        cards_langchain_docs = [LangchainDocument(doc) for doc in all_cards_texts]
        self.langchaing_docs.extend(cards_langchain_docs)

        self.markdown_separators = [
            "\n#{1,6} ",
            "```\n",
            "\n\\*\\*\\*+\n",
            "\n---+\n",
            "\n___+\n",
            "\n\n",
            "\n",
            " ",
            "",
        ]

        self.docs_processed = self.split_documents()

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_path,
            multi_process=True,
            model_kwargs={"device": "cuda"},
            encode_kwargs={
                "normalize_embeddings": True
            },  # Set `True` for cosine similarity
        )

        self.vector_database = FAISS.from_documents(
            self.docs_processed,
            self.embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
        )

        self.reranker = RAGPretrainedModel.from_pretrained(self.reranker_model_path)

    def split_documents(self):
        """
        Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
        """

        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(self.embedding_model_path),
            chunk_size=self.chunk_size,
            chunk_overlap=int(self.chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
            separators=self.markdown_separators,
        )

        docs_processed = []
        for doc in self.langchaing_docs:
            docs_processed += text_splitter.split_documents([doc])

        # Remove duplicates
        unique_texts = {}
        docs_processed_unique = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)

        return docs_processed_unique

    def retrieve_context(
        self, question: str, num_retrieved_docs: int = 30, num_docs_final: int = 5
    ):
        relevant_docs = self.vector_database.similarity_search(
            query=question, k=num_retrieved_docs
        )
        relevant_docs = [
            doc.page_content for doc in relevant_docs
        ]  # Keep only the text

        if self.reranker:
            relevant_docs = self.reranker.rerank(
                question, relevant_docs, k=num_docs_final
            )
            relevant_docs = [doc["content"] for doc in relevant_docs]

        relevant_docs = relevant_docs[:num_docs_final]

        context = "\nExtracted documents:\n"
        context += "".join(
            [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)]
        )

        return context
