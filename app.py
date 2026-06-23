import os
import glob
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

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
    if not documents:
        print("No valid documents found in ./data.")
        return

    print(f"Loaded {len(documents)} document(s).")
    
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=400)
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} chunks.")

    print("Initializing embeddings and Chroma DB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize Chroma, storing data to disk in ./chroma_db
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

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
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

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
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
