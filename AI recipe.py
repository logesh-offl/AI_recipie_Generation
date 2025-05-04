from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_community.document_loaders import WebBaseLoader
import bs4
import json
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import streamlit as st
embeddings = HuggingFaceHubEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
#google/flan-t5-base
llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.3", model_kwargs={"temperature": 0.7, "max_length": 100})
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "Enter hugging face token"

def web_link(webpath):
    loader = WebBaseLoader(
    web_paths=(webpath,),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs1 = loader.load()
    embeddings = HuggingFaceHubEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs1,embeddings)
    return vector_store.as_retriever()

def question(retrievers,quest):
    prompt = PromptTemplate.from_template(
    "Based on the following context to answer the question in minimum 3 lines:\n\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.3", model_kwargs={"temperature": 0.7, "max_length": 100})
# Define the correct pipeline
    data = (
        {"context": retrievers | (lambda docs: "\n\n".join([doc.page_content for doc in docs])), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Invoke query
    response = data.invoke(quest)
    return response


st.tittle("RAG application")
st.markdown("Enter a **web link** to extract content and ask questions based on it!")


web_url = st.text_input("🌐 Enter a Web URL:", "https://www.analyticsvidhya.com/blog/2023/09/retrieval-augmented-generation-rag-in-ai/")

if st.button("🔄 Process Web Link"):
    if web_url:
        retriever = web_link(web_url)
        st.session_state["retriever"] = retriever
        st.success("✅ Web content loaded! Now ask a question.")

# User asks a question
if "retriever" in st.session_state:
    user_question = st.text_input("❓ Ask a Question:")
    
    if st.button("💡 Get Answer"):
        if user_question:
            answer = question(st.session_state["retriever"], user_question)
            st.write("### 📌 Answer:")
            st.info(answer)

