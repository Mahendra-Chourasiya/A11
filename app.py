import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain_community.chat_message_histories import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import json

# Ensure the pdfs folder exists
os.makedirs("pdfs", exist_ok=True)

# Set up embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Streamlit layout
st.set_page_config(layout="wide")
st.title("Conversational RAG With PDF Uploads and Chat History")

# Input the OpenAI API Key
openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")

# Sidebar - Session Management
st.sidebar.header("Session Management")
session_id = st.sidebar.text_input("Session ID", value="default_session")
chat_history_path = f"chat_history/{session_id}.json"

if not os.path.exists("chat_history"):
    os.makedirs("chat_history")

# Load chat history from JSON
def load_chat_history(session_id):
    chat_history = InMemoryChatMessageHistory()
    try:
        with open(f"chat_history/{session_id}.json", "r") as history_file:
            messages = json.load(history_file)
            for msg in messages:
                if msg['role'] == 'assistant':
                    chat_history.add_assistant_message(msg['content'])
                elif msg['role'] == 'user':
                    chat_history.add_user_message(msg['content'])
    except FileNotFoundError:
        pass  # No previous history found, return an empty history
    return chat_history

# Save chat history to JSON
def save_chat_history(session_id, chat_history):
    messages = [{"role": "assistant", "content": msg['content']} if msg['role'] == 'assistant' else {"role": "user", "content": msg['content']} for msg in chat_history.messages]
    with open(f"chat_history/{session_id}.json", "w") as history_file:
        json.dump(messages, history_file)

chat_history = load_chat_history(session_id)

# Display available sessions
st.sidebar.subheader("Saved Sessions")
saved_sessions = [f.replace(".json", "") for f in os.listdir("chat_history") if f.endswith(".json")]
for saved_session in saved_sessions:
    if st.sidebar.button(f"Load {saved_session}"):
        session_id = saved_session
        st.experimental_rerun()

# Main content (left column)
col1, col2 = st.columns([2, 1])

with col1:
    if openai_api_key:
        llm = OpenAI(api_key=openai_api_key)

        # Function to get session history
        def get_session_history(session_id):
            return chat_history

        # Chat interface
        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            session_history.add_user_message(user_input)

            # Load all PDFs from the "pdfs" folder
            pdf_files = sorted([os.path.join("pdfs", f) for f in os.listdir("pdfs") if f.endswith(".pdf")])

            if pdf_files:
                documents = []
                for pdf_file in pdf_files:
                    loader = PyPDFLoader(pdf_file)
                    docs = loader.load()
                    documents.extend(docs)

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
                splits = text_splitter.split_documents(documents)
                vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
                retriever = vectorstore.as_retriever()

                # System prompt for contextualizing the question
                contextualize_q_system_prompt = (
                    "Given a chat history and the latest user question "
                    "which might reference context in the chat history, "
                    "formulate a standalone question which can be understood "
                    "without the chat history. Do NOT answer the question, "
                    "just reformulate it if needed and otherwise return it as is."
                )
                contextualize_q_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", contextualize_q_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )

                history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

                # System prompt for answering the question
                system_prompt = (
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, say that you "
                    "don't know. Use three sentences maximum and keep the "
                    "answer concise."
                    "\n\n"
                    "{context}"
                )
                qa_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )

                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

                # Conversational RAG chain
                response = rag_chain.invoke({"input": user_input, "chat_history": session_history.messages})
                st.write("Assistant:", response['answer'])
                session_history.add_assistant_message(response['answer'])
                save_chat_history(session_id, session_history)
            else:
                st.warning("No PDFs available in the 'pdfs' folder.")
    else:
        st.warning("Please enter your OpenAI API key")

# Sidebar - PDF Management
with col2:
    st.header("Manage PDFs")
    uploaded_files = st.file_uploader("Add a PDF", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join("pdfs", uploaded_file.name)
            with open(file_path, "wb") as file:
                file.write(uploaded_file.getvalue())
        st.success(f"Uploaded {len(uploaded_files)} file(s) to the 'pdfs' folder.")

    st.subheader("Available PDFs")
    pdf_files = sorted([f for f in os.listdir("pdfs") if f.endswith(".pdf")])
    for pdf in pdf_files:
        if st.button(f"Remove {pdf}"):
            os.remove(os.path.join("pdfs", pdf))
            st.experimental_rerun()
        else:
            st.write(pdf)
