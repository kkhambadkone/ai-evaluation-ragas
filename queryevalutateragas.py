import chromadb
import ollama
from chromadb.utils import embedding_functions
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
import logging
import os
from ragas import evaluate, RunConfig

run_config = RunConfig(timeout=1200, max_retries=3, max_wait=60, max_workers=2)


os.environ["RAGAS_DO_NOT_TRACK"] = "true"

logging.basicConfig(level=logging.DEBUG)

# --- Set up local Ollama judge ---
llm = LangchainLLMWrapper(ChatOllama(model="mistral", 
                                     format="json",
                                     timeout=1200,      # increase request timeout
                                     max_retries=3 ))
embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="mistral"))

# --- Config ---
COLLECTION_NAME = "pdf_docs"
OLLAMA_MODEL = "mistral"

# --- Connect to ChromaDB ---
print("Loading embedding model...")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)
print(f"Collection has {collection.count()} chunks\n")


def retrieve(question: str, top_k: int = 5) -> list[str]:
    """Retrieve top_k chunks from ChromaDB."""
    results = collection.query(query_texts=[question], n_results=top_k)
    return results["documents"][0]


def generate(question: str, contexts: list[str]) -> str:
    """Generate an answer using Ollama with the retrieved context."""
    context_str = "\n\n".join(contexts)
    prompt = f"""Answer the question using only the context provided below.
If the answer is not in the context, say "I don't know."

Context:
{context_str}

Question: {question}
Answer:"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()


# --- Test dataset ---
# Add your own questions and ground truth answers here
test_cases = [
    {
        "question": "Give me information on Auroville",
        "ground_truth": "Your expected answer here"
    }
    #,
    #{
    #    "question": "What are the key findings?",
    #    "ground_truth": "Your expected answer here"
    #},
]

# --- Run RAG pipeline and collect results ---
questions, answers, contexts, ground_truths = [], [], [], []

for tc in test_cases:
    print(f"Processing: {tc['question']}")
    retrieved = retrieve(tc["question"])
    answer = generate(tc["question"], retrieved)

    questions.append(tc["question"])
    answers.append(answer)
    contexts.append(retrieved)          # list of strings per question
    ground_truths.append(tc["ground_truth"])

    print(f"Answer: {answer}\n")

# --- Evaluate with RAGAS ---
print("Evaluating with RAGAS...")

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=llm,
    embeddings=embeddings,
    run_config=run_config,
    raise_exceptions=True
)

print("\n--- RAGAS Scores ---")
print(results)
