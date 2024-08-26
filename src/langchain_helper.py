import os
import src
import pickle

import tqdm as notebook_tqdm
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.loggers import logger
from src.data_preparation import processing_files

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


def vectorize_data():
    logger.info("Started loading CSV files")
    csv_loader = CSVLoader(
        file_path = src.NEW_FILE_NAME, 
        source_column = "prompt",
        encoding="utf-8"
    )

    data = csv_loader.load()

    hf_embedding = HuggingFaceEmbeddings(
        model_name= src.MODEL_NAME,
        model_kwargs= src.MODEL_KWARGS,
        encode_kwargs= src.ENCODE_KWARGS
    )

    logger.info("Creating FAISS vectors for the given data")
    faiss_vectorstore_db = FAISS.from_documents(
        documents = data,
        embedding = hf_embedding
    )

    # logger.info("Saving FAISS vectors of the data locally")
    # faiss_vectorstore_db.save_local(src.VECTORDB_FILE_PATH)

    logger.info('Saved serialiezed vector index into a pickle file...')
    vector_index_serialized = faiss_vectorstore_db.serialize_to_bytes()
    with open(src.VECTORDB_FILE_PATH, "wb+") as dump_file:
        pickle.dump(vector_index_serialized, dump_file)

    return hf_embedding


def format_docs(data):
    logger.info("Formatting documents page content")
    return "\n\n".join(doc.page_content for doc in data)


def get_qa_chain(hf_embedding: HuggingFaceEmbeddings):

    logger.info('Loading saved serialiezed vector index from pickle file...')
    if os.path.exists(src.VECTORDB_FILE_PATH):
        with open(src.VECTORDB_FILE_PATH, "rb+") as load_file:
            vector_index_serialized = pickle.load(load_file)

        logger.info('Deserializing the loaded serialized vectors...')
        faiss_vectorstore_db = FAISS.deserialize_from_bytes(
            serialized= vector_index_serialized,
            embeddings = hf_embedding,
            allow_dangerous_deserialization= True
        )

        logger.info("Creating a retriever for querying the vector database")
        faiss_retriever = faiss_vectorstore_db.as_retriever(search_kwargs={'score_threshold': 0.8})

        logger.info("Creating a prompt template for querying ")
        prompt_template = """Given the following context and a question, generate an answer based on this context only. 
        In the answer try to provide as much text as possible from "response" section in the source document context without 
        making much changes. If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

        CONTEXT: {context}

        QUESTION: {question}"""

        logger.info("Instantiation prompt template")
        customized_prompt = PromptTemplate(
            template = prompt_template,
            input_variables = ["context", "question"]
        )

        logger.info("Instatiating google gemini-pro model")
        google_llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.6) # google_api_key = GOOGLE_API_KEY

        logger.info("Instatiating Retrieval chain")
        retrieval_chain = (
            {
                "context": faiss_retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | customized_prompt
            | google_llm
            | StrOutputParser()
        )

        logger.info("Completed Retrieval chain execution")

    return retrieval_chain


if __name__ == "__main__":
    processed_files = processing_files()
    if processed_files:
        logger.info("Executed processing_files...")
    
    embeddings = vectorize_data()
    if embeddings:
        logger.info("FAISS Embeddings created...")
        retrieval_chain = get_qa_chain(embeddings)

    print(retrieval_chain.invoke("Do you have javascript course?"))