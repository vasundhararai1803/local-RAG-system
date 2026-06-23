import os
import glob
import uuid
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Load env variables
load_dotenv()

# Setup paths
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="Local RAG System", page_icon="🤖")

if "user_session_id" not in st.session_state:
    st.session_state.user_session_id = f"user_{uuid.uuid4().hex[:8]}"

@st.cache_resource(show_spinner=False)
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

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    import chromadb
    client = chromadb.PersistentClient(path="./chroma_db")
    collection_exists = False
    try:
        client.get_collection(session_id)
        collection_exists = True
    except Exception:
        pass

    if not documents and not collection_exists:
        return "NO_DOCS"

    if documents:
        # Split documents
        text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=90)
        splits = text_splitter.split_documents(documents)
        
        # Chroma DB
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory="./chroma_db",
            collection_name=session_id
        )
    else:
        # Load existing Chroma DB
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
            collection_name=session_id
        )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # Reranking Setup
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

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
    history_aware_retriever = create_history_aware_retriever(llm, compression_retriever, contextualize_q_prompt)

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

def process_uploaded_files(uploaded_files, session_id):
    if not uploaded_files:
        return
    with st.spinner("Processing documents..."):
        try:
            import chromadb
            client = chromadb.PersistentClient(path="./chroma_db")
            try:
                client.delete_collection(session_id)
            except Exception:
                pass
            # Clear previous files in data directory to prevent re-ingestion
            user_data_dir = os.path.join(DATA_DIR, session_id)
            if os.path.exists(user_data_dir):
                for old_file in glob.glob(f"{user_data_dir}/*"):
                    if os.path.isfile(old_file):
                        os.remove(old_file)
        except Exception as e:
            st.warning(f"Could not cleanly reset previous data: {e}")

        user_data_dir = os.path.join(DATA_DIR, session_id)
        os.makedirs(user_data_dir, exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = os.path.join(user_data_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Clear the cached chain so it reloads with new docs
        st.cache_resource.clear()
        st.toast("Documents processed successfully!", icon="✅")

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
            with st.expander("🔍 View Sources Cited"):
                for src in message["sources"]:
                    if isinstance(src, dict):
                        st.markdown(f"**{src['source']}**")
                        preview = src['content'][:300] + "..." if len(src['content']) > 300 else src['content']
                        st.info(preview)
                    else:
                        # Fallback for old history format
                        st.write(f"- {src}")

# React to user input
if prompt_data := st.chat_input("Ask a question about your documents...", accept_file="multiple"):
    
    # Extract files and text safely
    if hasattr(prompt_data, "files"):
        files = prompt_data.files
        prompt_text = prompt_data.text
    elif isinstance(prompt_data, dict):
        files = prompt_data.get("files", [])
        prompt_text = prompt_data.get("text", "")
    else:
        files = []
        prompt_text = prompt_data

    # Auto-process attached files
    if files:
        process_uploaded_files(files, st.session_state.user_session_id)
        
    if prompt_text:
        rag_chain = get_rag_chain(st.session_state.user_session_id)
        
        if rag_chain == "NO_DOCS":
            st.warning("⚠️ Please attach a PDF or text document first using the '+' button before asking questions!")
        else:
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt_text)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt_text})
    
            # Get response
            unique_sources = []
            
            with st.chat_message("assistant"):
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
                    router_result = router_chain.invoke({"input": prompt_text})
                    
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
                            for chunk in rag_chain.stream({"input": prompt_text, "chat_history": chat_history}):
                                if "context" in chunk:
                                    context_docs.extend(chunk["context"])
                                if "answer" in chunk:
                                    yield chunk["answer"]
                        
                        response = st.write_stream(generate_response())
                        
                        if context_docs:
                            unique_sources = []
                            seen_chunks = set()
                            for doc in context_docs:
                                source = doc.metadata.get("source", "Unknown Source")
                                page = doc.metadata.get("page")
                                src_str = f"{source} (Page {page})" if page is not None else f"{source}"
                                chunk_text = doc.page_content
                                if chunk_text not in seen_chunks:
                                    unique_sources.append({"source": src_str, "content": chunk_text})
                                    seen_chunks.add(chunk_text)
                    except Exception as e:
                        response = f"An error occurred: {e}"
                        st.markdown(response)

            if unique_sources:
                with st.expander("🔍 View Sources Cited"):
                    for src in unique_sources:
                        st.markdown(f"**{src['source']}**")
                        preview = src['content'][:300] + "..." if len(src['content']) > 300 else src['content']
                        st.info(preview)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response, "sources": unique_sources})
