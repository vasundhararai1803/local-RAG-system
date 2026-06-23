import os
import glob
import uuid
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Load env variables
load_dotenv()

# Setup paths
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="Local RAG System", page_icon="🤖")

if "user_session_id" not in st.session_state:
    st.session_state.user_session_id = f"user_{uuid.uuid4().hex[:8]}"

@st.cache_resource
def get_rag_chain(session_id):
    """Initializes and returns the RAG chain."""
    
    user_data_dir = os.path.join(DATA_DIR, session_id)
    os.makedirs(user_data_dir, exist_ok=True)

    # Load all documents from the directory
    documents = []
    for file_path in glob.glob(f"{user_data_dir}/**/*", recursive=True):
        if os.path.isfile(file_path):
            try:
                if file_path.endswith('.txt'):
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
                elif file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    if not documents:
        return "NO_DOCS"

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=400)
    splits = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Chroma DB
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory="./chroma_db",
        collection_name=session_id
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Language Model
    llm = ChatOllama(model="llama3.2", temperature=0.0, num_ctx=4096)

    # Contextualize question prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # System prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If the answer is not present in the context, explicitly say "
        "'I cannot find that in the documents'. Do not hallucinate or use outside knowledge.\n\n"
        "Strict Typography Rules:\n"
        "- Format main titles or questions using prominent subheaders (e.g., '## Question Title').\n"
        "- DO NOT use primary Markdown headers ('#' or '##') for procedural items like 'Step 1', 'Step 2', etc.\n"
        "- Format step breakdowns using clean bold text (e.g., '**Step 1: Heading Text**') or minor headers ('### Step 1').\n\n"
        "Context:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# --- Sidebar ---
with st.sidebar:
    st.title("📄 Document Upload")
    uploaded_files = st.file_uploader("Upload your PDF or TXT files here", type=["pdf", "txt"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    import chromadb
                    client = chromadb.PersistentClient(path="./chroma_db")
                    try:
                        client.delete_collection(st.session_state.user_session_id)
                    except Exception:
                        pass
                    # Clear previous files in data directory to prevent re-ingestion
                    user_data_dir = os.path.join(DATA_DIR, st.session_state.user_session_id)
                    if os.path.exists(user_data_dir):
                        for old_file in glob.glob(f"{user_data_dir}/*"):
                            if os.path.isfile(old_file):
                                os.remove(old_file)
                except Exception as e:
                    st.warning(f"Could not cleanly reset previous data: {e}")

                user_data_dir = os.path.join(DATA_DIR, st.session_state.user_session_id)
                os.makedirs(user_data_dir, exist_ok=True)
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(user_data_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Clear the cached chain so it reloads with new docs
                st.cache_resource.clear()
                st.success("Documents uploaded and processed successfully!")

# --- Main App ---
st.title("🤖 Local RAG System")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📌 View Source References"):
                for src in message["sources"]:
                    st.write(f"- {src}")

# React to user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response
    unique_sources = []
    
    with st.chat_message("assistant"):
        rag_chain = get_rag_chain(st.session_state.user_session_id)
        
        if rag_chain == "NO_DOCS":
            response = "I couldn't find any documents. Please upload some files in the sidebar first."
            st.markdown(response)
        else:
            with st.spinner("Thinking..."):
                # Routing Guardrail
                llm_router = ChatOllama(model="llama3.2", temperature=0.0, num_ctx=4096)
                router_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a router. Analyze the user query. "
                               "If it is completely unrelated to academics, grading, or the uploaded documents (e.g. asking for a baking recipe, code execution, general knowledge), output 'OFF_TOPIC'. "
                               "Otherwise, output 'RELEVANT'. Only output exactly one of these words."),
                    ("human", "{input}")
                ])
                router_chain = router_prompt | llm_router
                router_result = router_chain.invoke({"input": prompt})
                
            if "OFF_TOPIC" in router_result.content:
                response = "I am an academic document assistant and cannot answer off-topic queries."
                st.markdown(response)
            else:
                try:
                    # Construct chat history for the chain
                    chat_history = []
                    for msg in st.session_state.messages[:-1]: # Exclude the current prompt we just appended
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        else:
                            chat_history.append(AIMessage(content=msg["content"]))
                            
                    context_docs = []
                    
                    def generate_response():
                        for chunk in rag_chain.stream({"input": prompt, "chat_history": chat_history}):
                            if "context" in chunk:
                                context_docs.extend(chunk["context"])
                            if "answer" in chunk:
                                yield chunk["answer"]
                    
                    response = st.write_stream(generate_response())
                    
                    if context_docs:
                        sources = []
                        for doc in context_docs:
                            source = doc.metadata.get("source", "Unknown Source")
                            page = doc.metadata.get("page")
                            if page is not None:
                                sources.append(f"{source} (Page {page})")
                            else:
                                sources.append(f"{source}")
                        # Deduplicate while preserving order
                        unique_sources = list(dict.fromkeys(sources))
                except Exception as e:
                    response = f"An error occurred: {e}"
                    st.markdown(response)

        if unique_sources:
            with st.expander("📌 View Source References"):
                for src in unique_sources:
                    st.write(f"- {src}")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response, "sources": unique_sources})
