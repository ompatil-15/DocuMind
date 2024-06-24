import os
import shutil
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_pdf_text(pdf_docs):
    text = "AI"
    docs = load_document_info()
    
    for pdf in pdf_docs:
        if pdf not in docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, vs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_texts(text_chunks, embedding=embeddings)
    db.save_local(vs)
    print(vs)
    return db   

def save_document_info(document_names):
    with open("docs.txt", "w") as file:
        for name in document_names:
            file.write(name + "\n")

def load_document_info():
    document_names = []
    with open("docs.txt", "r") as file:
        for line in file:
            document_names.append(line.strip())
    return document_names

def clear_document_info():
    with open("docs.txt", "w") as file:
        file.truncate(0)
    if os.path.exists("vector_db/faiss_index"):
        shutil.rmtree("vector_db/faiss_index")
    os.makedirs("vector_db/faiss_index")

def create_chain(db):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    retriever = db.as_retriever()

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    qa_system_prompt = """
    You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer concise.\
    {context}
    """
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    print("Chain created")
    
    return conversational_rag_chain

def main():
    
    flag = 1
    if 'init' not in st.session_state:
        st.session_state['init'] = False
    
    
    st.set_page_config(
        page_title="DocuMind: Your Lightning-Fast⚡ Document📄 Assistant🤖",
        page_icon="📄",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    document_names = load_document_info()
    
    with st.sidebar:
        st.title("DocuMind ⚡📄🤖")
        st.markdown("---")
        
        st.subheader('Model Mode')
        modes = ['Regular', 'Plant Expert', 'Legal Assistant', 'Finance Master']
        if 'previous_mode' not in st.session_state:
            st.session_state.previous_mode = st.session_state.get('current_mode', 'Regular') 
            
        st.session_state.previous_mode = st.session_state.get('mode', 'Regular')

        mode = st.sidebar.selectbox('Choose Pre-Trained AI', modes, key='st.session_state.mode')
        
        
        st.session_state.mode = mode
        
        print("Previous Mode - ", st.session_state.previous_mode)
        if st.session_state.mode == "Regular":
            flag = 0
            print("Mode - Regular")
            vstore = "vector_db/faiss_index"
            welcome_message = "Upload a document to get started and ask any questions related to its content!"
            if st.session_state.previous_mode != st.session_state.mode:
                    st.session_state.init = False
                    st.session_state.pop('messages', None)
                
        if st.session_state.mode == "Plant Expert":
            flag = 0
            print("Mode - Plant expert")
            vstore = "vector_db/faiss_index_1"
            welcome_message = "Ask me questions about permaculture!"
            if st.session_state.previous_mode != st.session_state.mode:
                st.session_state.init = False
                st.session_state.pop('messages', None)          

        if st.session_state.mode == "Legal Assistant":
            flag = 0
            print("Mode - Legal assistant")
            vstore = "vector_db/faiss_index_2"
            welcome_message = "Ask me about the legal system of India!"
            if st.session_state.previous_mode != st.session_state.mode:
                st.session_state.init = False
                st.session_state.pop('messages', None)
            
        if st.session_state.mode == "Finance Master":
            flag = 0
            print("Mode - Finance master")
            vstore = "vector_db/faiss_index_3"
            welcome_message = "Ask me anything about the world of finance!" 
            if st.session_state.previous_mode != st.session_state.mode:
                st.session_state.init = False
                st.session_state.pop('messages', None)
          
        if st.session_state.mode == modes[0]:       
            st.markdown("### Upload Document")
            uploaded_files = st.file_uploader("Choose a document", type=["pdf", "docx", "txt"], accept_multiple_files=True)
            state = st.button("Submit")
            if state or st.session_state.previous_mode != st.session_state.mode:
                st.session_state.previous_mode = st.session_state.mode
                print("Processing")
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(uploaded_files)
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.db = get_vector_store(text_chunks, vstore)
                    st.success("Done")                
                    document_names.extend([file.name for file in uploaded_files])
                    save_document_info(document_names)
        
        # if current_mode != mode:
        #     print("Mode Changed")
        #     st.session_state.previous_mode = mode
        #     st.experimental_rerun()
        print(flag, st.session_state.init)
        if flag or not st.session_state.init:
            st.session_state.init = True
            try:
                st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", batch_size=90)
                print("Embeddings model loaded.")
            except Exception as e:
                st.error("An error occurred while creating embeddings.")
                st.exception(e)

            try:
                st.session_state.db = FAISS.load_local(vstore, st.session_state.embeddings, allow_dangerous_deserialization=True)
                print("Persistent directory: ", vstore)
                print("FAISS index successfully loaded.")
            except Exception as e:
                print("Creating empty embeddings")
                st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                st.session_state.db = FAISS.from_texts(" ", embedding=st.session_state.embeddings)
                st.session_state.db.save_local(vstore)

            try:
                st.session_state.chain = create_chain(st.session_state.db)
                print("Chain successfully created.")
            except Exception as e:
                st.error("An error occurred while creating the chain.")
                st.exception(e)
        
        if st.session_state.mode == modes[0]:       
        
            st.subheader("Your Documents")
            for document_name in document_names:
                st.write(document_name)
            if not document_names:
                st.markdown("*Uploaded documents will be listed here*")
        
            if st.button("Clear"):
                with st.spinner("Clearing..."):
                    clear_document_info()
                    document_names.clear()
                    print("Documents cleared")
                    st.experimental_rerun() 
        
        st.markdown("---")
        st.markdown("### About")
        st.info("DocuMind is your intelligent assistant for interacting with documents. Upload a document and ask questions to get insights quickly.")
    
    st.title("Welcome to DocuMind ⚡📄🤖")
    st.info("Your Lightning-Fast Document Assistant")
    
    if "messages" not in st.session_state.keys(): 
        st.session_state.messages = [
            {"role": "assistant", "content": welcome_message}
        ]
        
    user_input = st.chat_input("Message DocuMind", key="chatbox_input")    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
    for message in st.session_state.messages: 
        with st.chat_message(message["role"]):
            st.write(message["content"]) 
    
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                user_message = st.session_state.messages[-1]["content"]
                response = st.session_state.chain.invoke(
                    {"input": user_message},
                    config={
                        "configurable": {"session_id": "S001"}
                    })
                print("Chain history - ", response["chat_history"])
                st.write(response["answer"])
                message = {"role": "assistant", "content": response["answer"]}
                st.session_state.messages.append(message)
   
if __name__ == "__main__":
    main()
