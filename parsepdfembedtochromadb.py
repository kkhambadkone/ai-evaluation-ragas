import pymupdf4llm
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import MarkdownTextSplitter

# --- Config ---
PDF_PATH = "KSM-01-26.pdf"
COLLECTION_NAME = "pdf_docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Step 1: Parse PDF to Markdown ---
md_text = pymupdf4llm.to_markdown(PDF_PATH)

# --- Step 2: Chunk the Markdown ---
splitter = MarkdownTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
chunks = splitter.create_documents([md_text])
print(f"Created {len(chunks)} chunks")

# --- Step 3: Set up ChromaDB with local embedding model ---
client = chromadb.PersistentClient(path="./chroma_db")  # persists to disk

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"  # fast, good quality, runs locally
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}  # cosine similarity
)

# --- Step 4: Store chunks in ChromaDB ---
documents = [chunk.page_content for chunk in chunks]
ids = [f"chunk_{i}" for i in range(len(chunks))]
metadatas = [{"source": PDF_PATH, "chunk_index": i} for i in range(len(chunks))]

collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)
print(f"Stored {len(documents)} chunks in ChromaDB")

# --- Step 5: Query it ---
def query(question: str, top_k: int = 5):
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )
    print("======================RESULTS OF QUERY======================")
    for i, (doc, meta, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        print(f"\n[{i+1}] Score: {1 - distance:.4f} | Chunk: {meta['chunk_index']}")
        print(doc)

query("What is this document about?")
