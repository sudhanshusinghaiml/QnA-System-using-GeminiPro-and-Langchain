import streamlit as st
from src.data_preparation import processing_files
from src.langchain_helper import vectorize_data, get_qa_chain
from src.loggers import logger

st.title("Question & Answers chatbot 🌱")

clicked_button = st.button("Create Knowledgebase")

if clicked_button:
    processed_files = processing_files()
    if processed_files:
        logger.info("Executed processing_files...")

question = st.text_input("Question: ")
submit_query = st.button("Submit Question")

if submit_query:
    logger.info(f"Query: {question}")
    embeddings = vectorize_data()
    if embeddings:
        logger.info("FAISS Embeddings created")
        retrieval_chain = get_qa_chain(embeddings)
        response = retrieval_chain.invoke(question)
        st.header("Answer")
        st.write(response)
    else:
        logger.exception("Embeddings not found...")
