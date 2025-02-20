import os
import streamlit as st
import pickle
import time
import langchain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from huggingface_hub import login


from langchain_community.llms import HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="bigscience/bloom",
    task="text-generation",
    temperature=0.6
)

from dotenv import load_dotenv

load_dotenv()

st.title("News Research Tool📰")
st.sidebar.title("News Research URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URL")
file_path = "faiss_store_huggingface.pkl"

main_placeholder = st.empty()

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...📊")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitting...Started...📊")
    docs = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings()
    vectorstore_huggingface = FAISS.from_documents(docs, embeddings)

    main_placeholder.text("Embeddings Vector Started Building ....📊")
    time.sleep(2)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_huggingface, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
          vectorstore =  pickle.load(f)
          chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
          result = chain({"question": query}, return_only_outputs=True)
          #{"answer":  '" ", "sources": [] }
          st.header("Answer")
          st.write(result["answer"])

          sources = result.get("sources", "")
          if sources:
              st.subheader("Sources:")
              sources_list = sources.split("\n")
              for source in sources_list:
                  st.write(source)
