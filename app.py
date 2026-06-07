from dotenv import load_dotenv
load_dotenv()

import os
import chainlit as cl

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ==========================
# CHAT START
# ==========================

@cl.on_chat_start
async def start():
    api_key = os.getenv("GOOGLE_API_KEY")
    # Ask user to upload PDF
    files = await cl.AskFileMessage(
        content=f"""Please upload a PDF file and api key is : {api_key[:5]}""",
        accept=["application/pdf"],
        max_size_mb=30
    ).send()

    file = files[0]

    await cl.Message(
        content=f"Processing {file.name}..."
    ).send()

    # ==========================
    # LOAD PDF
    # ==========================

    loader = PyPDFLoader(file.path)
    documents = loader.load()

    # ==========================
    # SPLIT DOCUMENT
    # ==========================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)


    #==================================
    # VERIFICATION API IS LOADED OR NOT
    #==================================

    api_key = os.getenv("GOOGLE_API_KEY")

    print("API Key Found:", api_key is not None)

    if api_key:
        print("API Key Length:", len(api_key))
        print("API Key Starts With:", api_key[:5])

    # ==========================
    # EMBEDDINGS
    # ==========================

    # embeddings = GoogleGenerativeAIEmbeddings(
    #     model="models/embedding-001"
    # )
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    # google_api_key="GOOGLE_API_KEY"
)


    # ==========================
    # VECTOR STORE
    # ==========================

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory= None
        # persist_directory="./chroma_db",
        
    )

    retriever = vectorstore.as_retriever()

    # ==========================
    # LLM
    # ==========================

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    # ==========================
    # PROMPT
    # ==========================

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""
    )

    # ==========================
    # FORMAT RETRIEVED DOCS
    # ==========================

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    # ==========================
    # RAG CHAIN
    # ==========================

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Save chain in session

    cl.user_session.set(
        "rag_chain",
        rag_chain
    )

    await cl.Message(
        content="PDF processed successfully. Ask your questions."
    ).send()


# ==========================
# HANDLE USER QUESTIONS
# ==========================

@cl.on_message
async def main(message: cl.Message):

    rag_chain = cl.user_session.get(
        "rag_chain"
    )

    if rag_chain is None:

        await cl.Message(
            content="Please upload a PDF first."
        ).send()

        return

    response = rag_chain.invoke(
        message.content
    )

    await cl.Message(
        content=response
    ).send()
