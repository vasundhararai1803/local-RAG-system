import os
import glob
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def load_documents(directory_path):
    documents = []
    # Search for files recursively in the directory
    for file_path in glob.glob(f"{directory_path}/**/*", recursive=True):
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
    return documents

def main():
    load_dotenv()

    print("Loading documents from ./data...")
    documents = load_documents("./data")
    print("Initializing embeddings and Chroma DB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    import chromadb
    client = chromadb.PersistentClient(path="./chroma_db")
    collection_exists = False
    try:
        client.get_collection('langchain')
        collection_exists = True
    except Exception:
        pass

    if not documents and not collection_exists:
        print("No valid documents found in ./data and no existing database found.")
        print("Please add documents to ./data first.")
        return

    if documents:
        print(f"Loaded {len(documents)} document(s).")
        print("Splitting documents into chunks...")
        text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=90)
        splits = text_splitter.split_documents(documents)
        print(f"Created {len(splits)} chunks.")

        # Initialize Chroma, storing data to disk in ./chroma_db
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory="./chroma_db"
        )
    else:
        print("Loading existing Chroma database...")
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    print("Setting up reranker...")
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    print("Setting up retrieval chain...")
    llm = ChatOllama(model="llama3.2", temperature=0.0, num_ctx=4096)

    # Strict system prompt to prevent hallucinations
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

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(compression_retriever, question_answer_chain)

    print("\n" + "="*40)
    print("      RAG System Initialized")
    print("="*40)
    print("Type your questions below. Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("\nQuestion: ")
            if user_input.lower().strip() == 'exit':
                print("Exiting...")
                break
            if not user_input.strip():
                continue
            
            response = rag_chain.invoke({"input": user_input})
            print(f"\nAnswer: {response['answer']}")
            
            if "context" in response and response["context"]:
                print("\n--- Sources Cited ---")
                for i, doc in enumerate(response["context"], 1):
                    source = doc.metadata.get("source", "Unknown Source")
                    page = doc.metadata.get("page")
                    src_str = f"{source} (Page {page})" if page is not None else f"{source}"
                    preview = doc.page_content[:150].replace("\n", " ") + "..."
                    print(f"{i}. {src_str}")
                    print(f"   Preview: {preview}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
