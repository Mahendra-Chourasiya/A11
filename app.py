# import streamlit as st
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.vectorstores import FAISS
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_openai import OpenAI
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# import os
# import pdfplumber

# # Ensure the pdfs folder exists
# os.makedirs("pdfs", exist_ok=True)

# # Set up embeddings
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # Set up Streamlit layout
# st.set_page_config(layout="centered", page_icon="🤖", page_title="Conversational RAG")
# st.markdown("<h1 style='text-align: center;'>Conversational RAG with PDF Support</h1>", unsafe_allow_html=True)
# st.write("<p style='text-align: center;'>Upload PDFs and chat with their content</p>", unsafe_allow_html=True)

# # Input the OpenAI and LangChain API Keys
# st.sidebar.title("🔑 API Keys")
# openai_api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
# langchain_api_key = st.sidebar.text_input("Enter your LangChain API key:", type="password")

# if openai_api_key and langchain_api_key:
#     llm = OpenAI(api_key=openai_api_key)

#     # Sidebar: PDF management
#     st.sidebar.title("📄 Document Manager")
#     uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             file_path = os.path.join("pdfs", uploaded_file.name)
#             with open(file_path, "wb") as file:
#                 file.write(uploaded_file.getvalue())
#         st.sidebar.success(f"Uploaded {len(uploaded_files)} file(s) to the 'pdfs' folder.")

#     # Display and download PDFs alphabetically
#     pdf_files_sorted = sorted(os.listdir("pdfs"))
#     for pdf in pdf_files_sorted:
#         file_path = os.path.join("pdfs", pdf)
#         with open(file_path, "rb") as file:
#             st.sidebar.download_button(
#                 label=f"📥 Download {pdf}",
#                 data=file,
#                 file_name=pdf,
#                 mime="application/pdf"
#             )

#     # Remove PDFs from the folder
#     st.sidebar.subheader("🗑️ Remove a PDF")
#     pdf_to_remove = st.sidebar.selectbox("Select a PDF to remove", pdf_files_sorted)
#     if st.sidebar.button("Remove PDF"):
#         os.remove(os.path.join("pdfs", pdf_to_remove))
#         st.sidebar.success(f"Removed {pdf_to_remove}")

#     # Chat interface
#     st.sidebar.title("💬 Chat Session")
#     session_id = st.sidebar.text_input("Session ID", value="default_session")
#     collaborator_id = st.sidebar.text_input("Collaborator ID", value="user1")

#     # Function to get session history
#     def get_session_history(session: str) -> BaseChatMessageHistory:
#         if session not in st.session_state:
#             st.session_state[session] = ChatMessageHistory()
#         return st.session_state[session]

#     # Load all PDFs from the "pdfs" folder
#     pdf_files = sorted([os.path.join("pdfs", f) for f in os.listdir("pdfs") if f.endswith(".pdf")])

#     if pdf_files:
#         documents = []
#         for pdf_file in pdf_files:
#             loader = PyPDFLoader(pdf_file)
#             docs = loader.load()
#             documents.extend(docs)

#         # Split and create embeddings for the documents
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
#         splits = text_splitter.split_documents(documents)
#         vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
#         retriever = vectorstore.as_retriever()

#         # System prompt for contextualizing the question
#         contextualize_q_system_prompt = (
#             "Given a chat history and the latest user question "
#             "which might reference context in the chat history, "
#             "formulate a standalone question which can be understood "
#             "without the chat history. Do NOT answer the question, "
#             "just reformulate it if needed and otherwise return it as is."
#         )
#         contextualize_q_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", contextualize_q_system_prompt),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}"),
#             ]
#         )

#         history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

#         # System prompt for answering the question
#         system_prompt = (
#             "You are an assistant for question-answering tasks. "
#             "Use the following pieces of retrieved context to answer "
#             "the question. If you don't know the answer, say that you "
#             "don't know. Use three sentences maximum and keep the "
#             "answer concise."
#             "\n\n"
#             "{context}"
#         )
#         qa_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", system_prompt),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}"),
#             ]
#         )

#         question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
#         rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

#         conversational_rag_chain = rag_chain

#         user_input = st.text_input("Your question:")
#         if user_input:
#             try:
#                 session_history = get_session_history(session_id)
#                 response = conversational_rag_chain.invoke(
#                     {
#                         "input": user_input,
#                         "chat_history": session_history.messages,  # Pass chat history here
#                         "context": "PDF Content or additional context here"
#                     }
#                 )
#                 st.write("Assistant:", response['answer'])
#                 st.write("Chat History:", session_history.messages)
#             except Exception as e:
#                 st.error(f"An error occurred: {str(e)}")
#     else:
#         st.warning("No PDFs available in the 'pdfs' folder.")
# else:
#     st.warning("Please enter both the OpenAI and LangChain API keys")





# import streamlit as st
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.vectorstores import FAISS
# from langchain.memory import ChatMessageHistory  # Import from langchain.memory
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain import OpenAI
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader
# import os

# # Ensure the pdfs folder exists
# os.makedirs("pdfs", exist_ok=True)

# # Set up embeddings
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # Set up Streamlit layout
# st.set_page_config(layout="centered", page_icon="🤖", page_title="Conversational RAG")
# st.markdown("<h1 style='text-align: center;'>Conversational RAG with PDF Support</h1>", unsafe_allow_html=True)
# st.write("<p style='text-align: center;'>Upload PDFs and chat with their content</p>", unsafe_allow_html=True)

# # Input the OpenAI and LangChain API Keys
# st.sidebar.title("🔑 API Keys")
# openai_api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
# langchain_api_key = st.sidebar.text_input("Enter your LangChain API key:", type="password")

# if openai_api_key and langchain_api_key:
#     llm = OpenAI(api_key=openai_api_key)

#     # Sidebar: PDF management
#     st.sidebar.title("📄 Document Manager")
#     uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             file_path = os.path.join("pdfs", uploaded_file.name)
#             with open(file_path, "wb") as file:
#                 file.write(uploaded_file.getvalue())
#         st.sidebar.success(f"Uploaded {len(uploaded_files)} file(s) to the 'pdfs' folder.")

#     # Display and download PDFs alphabetically
#     pdf_files_sorted = sorted(os.listdir("pdfs"))
#     for pdf in pdf_files_sorted:
#         file_path = os.path.join("pdfs", pdf)
#         with open(file_path, "rb") as file:
#             st.sidebar.download_button(
#                 label=f"📥 Download {pdf}",
#                 data=file,
#                 file_name=pdf,
#                 mime="application/pdf"
#             )

#     # Remove PDFs from the folder
#     st.sidebar.subheader("🗑️ Remove a PDF")
#     pdf_to_remove = st.sidebar.selectbox("Select a PDF to remove", pdf_files_sorted)
#     if st.sidebar.button("Remove PDF"):
#         os.remove(os.path.join("pdfs", pdf_to_remove))
#         st.sidebar.success(f"Removed {pdf_to_remove}")

#     # Chat interface
#     st.sidebar.title("💬 Chat Session")
#     session_id = st.sidebar.text_input("Session ID", value="default_session")
#     collaborator_id = st.sidebar.text_input("Collaborator ID", value="user1")

#     # Function to get session history
#     def get_session_history(session: str) -> ChatMessageHistory:
#         if session not in st.session_state:
#             st.session_state[session] = ChatMessageHistory()
#         return st.session_state[session]

#     # Load all PDFs from the "pdfs" folder
#     pdf_files = sorted([os.path.join("pdfs", f) for f in os.listdir("pdfs") if f.endswith(".pdf")])

#     if pdf_files:
#         documents = []
#         for pdf_file in pdf_files:
#             loader = PyPDFLoader(pdf_file)
#             docs = loader.load()
#             documents.extend(docs)

#         # Split and create embeddings for the documents
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
#         splits = text_splitter.split_documents(documents)
#         vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
#         retriever = vectorstore.as_retriever()

#         # System prompt for contextualizing the question
#         contextualize_q_system_prompt = (
#             "Given a chat history and the latest user question "
#             "which might reference context in the chat history, "
#             "formulate a standalone question which can be understood "
#             "without the chat history. Do NOT answer the question, "
#             "just reformulate it if needed and otherwise return it as is."
#         )
#         contextualize_q_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", contextualize_q_system_prompt),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}"),
#             ]
#         )

#         history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

#         # System prompt for answering the question
#         system_prompt = (
#             "You are an assistant for question-answering tasks. "
#             "Use the following pieces of retrieved context to answer "
#             "the question. If you don't know the answer, say that you "
#             "don't know. Use three sentences maximum and keep the "
#             "answer concise."
#             "\n\n"
#             "{context}"
#         )
#         qa_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", system_prompt),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}"),
#             ]
#         )

#         question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
#         rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

#         conversational_rag_chain = rag_chain

#         # Chat input and chat history display
#         session_history = get_session_history(session_id)

#         # Display chat history
#         for message in session_history.messages:
#             if message.type == "ai":
#                 st.chat_message("assistant").markdown(message.content)
#             elif message.type == "human":
#                 st.chat_message("user").markdown(message.content)

#         user_input = st.chat_input("Your question:")
#         if user_input:
#             try:
#                 # Update chat history
#                 session_history.add_user_message(user_input)
#                 response = conversational_rag_chain.invoke(
#                     {
#                         "input": user_input,
#                         "chat_history": session_history.messages,
#                         "context": "PDF Content or additional context here"
#                     }
#                 )
#                 # Display assistant response
#                 st.chat_message("assistant").markdown(response['answer'])
#                 session_history.add_ai_message(response['answer'])
#             except Exception as e:
#                 st.error(f"An error occurred: {str(e)}")
#     else:
#         st.warning("No PDFs available in the 'pdfs' folder.")
# else:
#     st.warning("Please enter both the OpenAI and LangChain API keys.")




import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain.memory import ChatMessageHistory  # Import from langchain.memory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os

# Ensure the pdfs folder exists and pre-uploaded PDFs are present
os.makedirs("pdfs", exist_ok=True)

# Pre-uploaded PDFs
pre_uploaded_pdfs = [
    "Generative_AI.pdf",
    "Moodle.pdf",
    "aws.pdf",
    "Drupal.pdf"
]

# Place these pre-uploaded PDFs in the 'pdfs' directory
# Make sure these files exist in the 'pdfs' directory before running the app
for pdf in pre_uploaded_pdfs:
    if not os.path.isfile(os.path.join("pdfs", pdf)):
        st.warning(f"Pre-uploaded file missing: {pdf}")

# Set up embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Streamlit layout
st.set_page_config(layout="centered", page_icon="🤖", page_title="Conversational RAG")
st.markdown("<h1 style='text-align: center;'>Conversational RAG with PDF Support</h1>", unsafe_allow_html=True)
st.write("<p style='text-align: center;'>Upload PDFs and chat with their content</p>", unsafe_allow_html=True)

# Input the OpenAI and LangChain API Keys
st.sidebar.title("🔑 API Keys")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
langchain_api_key = st.sidebar.text_input("Enter your LangChain API key:", type="password")

if openai_api_key and langchain_api_key:
    llm = OpenAI(api_key=openai_api_key)

    # Sidebar: PDF management
    st.sidebar.title("📄 Document Manager")
    uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join("pdfs", uploaded_file.name)
            with open(file_path, "wb") as file:
                file.write(uploaded_file.getvalue())
        st.sidebar.success(f"Uploaded {len(uploaded_files)} file(s) to the 'pdfs' folder.")

    # Display and download PDFs alphabetically
    pdf_files_sorted = sorted(os.listdir("pdfs"))
    for pdf in pdf_files_sorted:
        file_path = os.path.join("pdfs", pdf)
        with open(file_path, "rb") as file:
            st.sidebar.download_button(
                label=f"📥 Download {pdf}",
                data=file,
                file_name=pdf,
                mime="application/pdf"
            )

    # Remove PDFs from the folder
    st.sidebar.subheader("🗑️ Remove a PDF")
    pdf_to_remove = st.sidebar.selectbox("Select a PDF to remove", pdf_files_sorted)
    if st.sidebar.button("Remove PDF"):
        os.remove(os.path.join("pdfs", pdf_to_remove))
        st.sidebar.success(f"Removed {pdf_to_remove}")

    # Chat interface
    st.sidebar.title("💬 Chat Session")
    session_id = st.sidebar.text_input("Session ID", value="default_session")
    collaborator_id = st.sidebar.text_input("Collaborator ID", value="user1")

    # Function to get session history
    def get_session_history(session: str) -> ChatMessageHistory:
        if session not in st.session_state:
            st.session_state[session] = ChatMessageHistory()
        return st.session_state[session]

    # Load all PDFs from the "pdfs" folder
    pdf_files = sorted([os.path.join("pdfs", f) for f in os.listdir("pdfs") if f.endswith(".pdf")])

    if pdf_files:
        documents = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            documents.extend(docs)

        # Split and create embeddings for the documents
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

        conversational_rag_chain = rag_chain

        # Chat input and chat history display
        session_history = get_session_history(session_id)

        # Display chat history
        for message in session_history.messages:
            if message.type == "ai":
                st.chat_message("assistant").markdown(message.content)
            elif message.type == "human":
                st.chat_message("user").markdown(message.content)

        user_input = st.chat_input("Your question:")
        if user_input:
            try:
                # Update chat history
                session_history.add_user_message(user_input)
                response = conversational_rag_chain.invoke(
                    {
                        "input": user_input,
                        "chat_history": session_history.messages,
                        "context": "PDF Content or additional context here"
                    }
                )
                # Display assistant response
                st.chat_message("assistant").markdown(response['answer'])
                session_history.add_ai_message(response['answer'])
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("No PDFs available in the 'pdfs' folder.")
else:
    st.warning("Please enter both the OpenAI and LangChain API keys.")
