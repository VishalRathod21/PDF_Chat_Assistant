## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile

from dotenv import load_dotenv
load_dotenv()

# Initialize embeddings
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
if huggingface_api_key:
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_api_key

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={
        'device': 'cpu',  # Force CPU usage
        'trust_remote_code': True
    },
    encode_kwargs={'normalize_embeddings': True}
)


# Set page config
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4f46e5;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #4338ca;
        transform: translateY(-1px);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        display: flex;
        max-width: 80%;
    }
    .user-message {
        background-color: #e0e7ff;
        margin-left: auto;
        border-bottom-right-radius: 0.25rem;
    }
    .assistant-message {
        background-color: #f3f4f6;
        margin-right: auto;
        border-bottom-left-radius: 0.25rem;
    }
    .stTextInput>div>div>input {
        border-radius: 0.75rem;
        padding: 0.75rem 1rem;
        border: 1px solid #e5e7eb;
    }
    .stTextInput>div>div>input:focus {
        border-color: #4f46e5;
        box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/documents.png", width=80)
    st.markdown("## 🔑 Settings")
    
    with st.expander("⚙️ API & Model Settings", expanded=True):
        api_key = st.text_input("Enter your Groq API key:", type="password", 
                              help="Get your API key from https://console.groq.com/")
        
        model_name = st.selectbox(
            "Select Model",
            ["Gemma2-9b-It", "Llama3-8b-8192", "Mixtral-8x7b-32768"],
            index=0
        )
        
        session_id = st.text_input(
            "Session ID", 
            value="default_session",
            help="Use different IDs for separate conversations"
        )
    
    with st.expander("📄 Upload Documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Choose PDF files", 
            type="pdf", 
            accept_multiple_files=True,
            help="Upload one or more PDF files to chat with"
        )
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This is a modern RAG (Retrieval-Augmented Generation) application that lets you chat with your PDF documents.
    
    **How to use:**
    1. Enter your Groq API key
    2. Upload PDF files
    3. Start chatting with the AI about your documents
    """)

# Main content
st.title("📄 PDF Chat Assistant")
st.markdown("Ask questions about your uploaded PDF documents")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Check if Groq API key is provided
if not api_key:
    st.warning("Please enter your Groq API key in the sidebar to continue.")
    st.stop()

# Initialize the language model
llm = ChatGroq(
    groq_api_key=api_key,
    model_name=model_name,
    temperature=0.7,
    max_tokens=4000
)
    
# Initialize session state for chat history
if 'store' not in st.session_state:
    st.session_state.store = {}
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Process uploaded PDFs
if not uploaded_files:
    st.info("Please upload one or more PDF files to start chatting.")
    st.stop()

# Process uploaded PDFs
with st.spinner("Processing documents..."):
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        try:
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = uploaded_file.name
                doc.metadata['page'] = doc.metadata.get('page', 0) + 1
            documents.extend(docs)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    if not documents:
        st.error("No valid documents found. Please check your PDF files.")
        st.stop()
    
    # Process documents in chunks to avoid memory issues
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=300,
        length_function=len,
        is_separator_regex=False,
    )
    
    splits = text_splitter.split_documents(documents)
        
    # Create vector store
    with st.spinner("Creating search index..."):
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=None
        )
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10}
    )
        
    # Contextualize question with chat history
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, 
        retriever, 
        contextualize_q_prompt
    )
    
    # Enhanced system prompt for better responses
    system_prompt = """You are an expert assistant that helps users understand documents. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer concise but informative, using 3-5 sentences maximum. \
    Always cite the source document name when possible.
    \n\nContext:\n{context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
        
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(
        history_aware_retriever, 
        question_answer_chain
    )
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]
    
    st.session_state.conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    st.success(f"Processed {len(documents)} pages from {len(uploaded_files)} files")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Prepare the assistant's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                with st.spinner('Analyzing documents...'):
                    # Get response from the RAG chain
                    response = st.session_state.conversational_rag_chain.invoke(
                        {"input": prompt},
                        config={"configurable": {"session_id": session_id}},
                    )
                    full_response = response["answer"]
                
                # Display the response with markdown formatting
                message_placeholder.markdown(full_response)
                
                # Update chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Add a clear chat button
    if st.sidebar.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.store = {}
        st.rerun()

# Add a download chat history button
if st.session_state.messages:
    chat_history = "\n\n".join(
        f"**{msg['role'].capitalize()}:** {msg['content']}" 
        for msg in st.session_state.messages
    )
    st.sidebar.download_button(
        label="💾 Save Chat",
        data=chat_history,
        file_name=f"chat_history_{session_id}.md",
        mime="text/markdown"
    )
else:
    st.warning("Please enter your Groq API key in the sidebar to continue.")
