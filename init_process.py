

CHAT = "gpt-4o-mini"
emb_model = "text-embedding-3-small"
VECTOR_DIMENSIONS = 1536
top_k = 5

chunks = None

debug = False
delete = False



from azure.search.documents import SearchClient
from openai import AzureOpenAI
import time
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_core.messages import HumanMessage
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from tqdm import tqdm
import re
import json
import random
from dotenv import load_dotenv
import os

load_dotenv()

search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_index = os.getenv("AZURE_SEARCH_INDEX")
search_api_key = os.getenv("AZURE_SEARCH_KEY")

service_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
service_api_key = os.getenv("AZURE_OPENAI_KEY")

if not all([search_endpoint, search_index, search_api_key, service_endpoint, service_api_key]):
    raise ValueError("Missing environment variables. Check your .env file.")




load_dotenv()

my_llm = AzureChatOpenAI(
    azure_endpoint=service_endpoint,
    api_key=service_api_key,
    api_version="2024-10-01-preview",
    azure_deployment=CHAT,
    temperature=0.0,
)

openai_client = AzureOpenAI(
    azure_endpoint=service_endpoint,
    api_key=service_api_key,
    api_version="2024-10-01-preview",
)

search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=search_index,
    credential=AzureKeyCredential(search_api_key),
)


def is_index_empty():
    try:
        return search_client.get_document_count() == 0
    except Exception:
        return True


def create_embeddings(text):
    response = openai_client.embeddings.create(
        model=emb_model,
        input=text
    )
    return [d.embedding for d in response.data]

def remove_headers_footers(pages):
    cleaned_pages = []

    for page in pages:
        lines = page.page_content.splitlines()
        cleaned = []

        for line in lines:
            stripped = line.strip()

            # Remove page number patterns
            if re.match(r"^(P\s*a\s*g\s*e\s*)?\d+\s*(\|\s*\d+)?$", stripped):
                continue
            
            # Remove "Page X of Y"
            if re.match(r"^Page\s+\d+\s+of\s+\d+$", stripped, re.IGNORECASE):
                continue

            # Remove lines that are ONLY digits
            if stripped.isdigit():
                continue

            cleaned.append(line)

        page.page_content = "\n".join(cleaned)
        cleaned_pages.append(page)

    return cleaned_pages

def remove_table_of_contents(text):
    lines = text.splitlines()
    cleaned = []
    toc_mode = False

    # Pattern for dotted TOC items
    dotted_pattern = re.compile(r".*\.{3,}\s*\d+$")

    for line in lines:
        stripped = line.strip()

        if stripped.lower() in ("table of contents", "contents"):
            toc_mode = True
            continue

        if toc_mode:
            # Remove dotted leader TOC lines
            if dotted_pattern.match(line):
                continue

            # Remove "Title  15"
            if re.match(r".+\s+\d+$", stripped):
                continue

            # Exit TOC mode when regular text starts
            if len(stripped.split()) > 8:
                toc_mode = False

        if not toc_mode:
            cleaned.append(line)

    return "\n".join(cleaned)


def remove_tables(text):
    cleaned_lines = []

    for line in text.splitlines():
        # pipe tables
        if line.count("|") >= 3:
            continue

       
        digit_ratio = sum(c.isdigit() for c in line) / max(len(line), 1)
        if digit_ratio > 0.4:
            continue

        # excessive spacing
        if re.search(r"\s{4,}", line):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def should_merge_with_llm(prev_text, next_text):
    prompt = f"""
You are segmenting an HR policy document into coherent sections.

Should the following two text segments belong to the SAME section?

Segment A:
---
{prev_text}
---

Segment B:
---
{next_text}
---

Rules:
- If Segment B continues explaining the same policy, answer YES.
- If Segment B introduces a new topic, new section, or new rule, answer NO.
- Answer ONLY YES or NO.
"""

    response = my_llm.invoke([HumanMessage(content=prompt)])
    answer = response.content.strip().upper()
    return answer.startswith("YES")
def llm_semantic_chunk(
    sentence_docs,
    max_chunk_tokens=600
):
    chunks = []
    current_chunk = []

    for doc in sentence_docs:

        text = doc.page_content.strip()

        if not current_chunk:
            current_chunk.append(text)
            continue

        current_text = " ".join(current_chunk)
        token_count = len(current_text.split())

        # Hard token limit safeguard
        if token_count > max_chunk_tokens:
            chunks.append(current_text)
            current_chunk = [text]
            continue

        # Ask LLM if we should merge
        merge = should_merge_with_llm(current_text[-800:], text)
        # ^ we only send last 800 chars to reduce cost

        if merge:
            current_chunk.append(text)
        else:
            chunks.append(current_text)
            current_chunk = [text]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def semantic_chunk(
    sentence_docs,
    sentence_embeddings,
    similarity_threshold=0.7,
    max_chunk_tokens=600
):
    chunks = []
    current_chunk = []
    current_embeddings = []

    for doc, emb in zip(sentence_docs, sentence_embeddings):

        if not current_chunk:
            current_chunk.append(doc)
            current_embeddings.append(emb)
            continue

        avg_embedding = np.mean(
            np.array(current_embeddings), axis=0
        ).reshape(1, -1)

        emb_np = np.array(emb).reshape(1, -1)
        similarity = cosine_similarity(avg_embedding, emb_np)[0][0]

        token_count = sum(len(d.page_content.split()) for d in current_chunk)

        if similarity < similarity_threshold or token_count > max_chunk_tokens:
            combined_text = " ".join(d.page_content for d in current_chunk)
            chunks.append(combined_text)
            current_chunk = [doc]
            current_embeddings = [emb]
        else:
            current_chunk.append(doc)
            current_embeddings.append(emb)

    if current_chunk:
        combined_text = " ".join(d.page_content for d in current_chunk)
        chunks.append(combined_text)

    return chunks


def create_index():
    index_client = SearchIndexClient(
        endpoint=search_endpoint,
        credential=AzureKeyCredential(search_api_key),
    )

    if delete:
        existing = [i.name for i in index_client.list_indexes()]
        if search_index in existing:
            index_client.delete_index(search_index)
            if debug:
                print(f"Index '{search_index}' deleted")

    if search_index in [i.name for i in index_client.list_indexes()]:
        if debug:
            print(f"Index '{search_index}' already exists")
        return

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),

        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchableField(name="summary", type=SearchFieldDataType.String),

        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=VECTOR_DIMENSIONS,
            vector_search_profile_name="vector-profile",
        ),

        SearchField(
            name="summary_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=VECTOR_DIMENSIONS,
            vector_search_profile_name="vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-config",
                kind="hnsw",
                parameters={"metric": "cosine"},
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="vector-profile",
                algorithm_configuration_name="hnsw-config"
            )
        ],
    )

    index = SearchIndex(
        name=search_index,
        fields=fields,
        vector_search=vector_search
    )

    index_client.create_index(index)
    if debug:
        print(f"Index '{search_index}' created successfully")


def generate_section_summary(section_text):
    prompt = f"""The following text is from an HR document:{section_text}Write a 1-2 sentence summary of what this section contains."""
    response = my_llm.invoke([HumanMessage(content=prompt)])
    return response.content


def upload_docs():
    global chunks

    if not is_index_empty():
        if debug:
            print("docs already uploaded")
        return

    BASE_DIR = Path(__file__).resolve().parent
    pdf_path = BASE_DIR / "docs" / "hr_faqs.pdf"

    pdf_loader = PyPDFLoader(str(pdf_path))
    pages = pdf_loader.load()

   
    pages = remove_headers_footers(pages)

    for page in pages:
        text = page.page_content
        text = remove_table_of_contents(text)
        text = remove_tables(text)

        # Normalize whitespace
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n\s+", "\n", text)

        page.page_content = text

    sentence_splitter = TokenTextSplitter(
        chunk_size=100,
        chunk_overlap=0
    )

    sentence_docs = sentence_splitter.split_documents(pages)
    sentences = [doc.page_content for doc in sentence_docs]

    if debug:
        print("Generating sentence embeddings...")

    sentence_embeddings = []
    for i in tqdm(range(0, len(sentences), 10)):
        sentence_embeddings.extend(create_embeddings(sentences[i:i+10]))

    if debug:
        print("Creating LLM-based semantic chunks...")

    semantic_chunks = llm_semantic_chunk(sentence_docs)
    chunks = semantic_chunks


    if debug:
        print("Generating summaries and embeddings...")

    documents = []
    for i, chunk_text in enumerate(tqdm(semantic_chunks)):

        summary = generate_section_summary(chunk_text)

        content_vector = create_embeddings([chunk_text])[0]
        summary_vector = create_embeddings([summary])[0]

        documents.append({
            "id": f"doc-{i}",
            "content": chunk_text,
            "summary": summary,
            "content_vector": content_vector,
            "summary_vector": summary_vector,
        })

    if debug:
        print("Uploading documents to Azure Search...")

    search_client.upload_documents(documents=documents)
    time.sleep(4)

    if debug:
        print("Upload complete!")


def get_chunks():
    return chunks


def create_eval_dataset(n_of_ex):

    semantic_chunks = get_chunks()
    print("Loaded", len(semantic_chunks), "semantic chunks")

    selected_chunks = random.sample(semantic_chunks, min(n_of_ex, len(semantic_chunks)))
    print("Using", len(selected_chunks), "chunks for evaluation dataset")

    EVAL_OUTPUT_PATH = Path("evaluations/eval_dataset.json")
    EVAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    def generate_deepeval_examples(chunks_subset):
        examples = []

        for chunk in chunks_subset:
            prompt = f"""
                You are creating evaluation questions for an HR RAG system.

                Based ONLY on the following context:
                ---
                {chunk}
                ---

                Generate exactly 1 useful employee-facing question that this context can answer.
                For that question, also write a one-sentence accurate answer.

                Return ONLY valid JSON in this exact format:
                [
                  {{"question": "...", "answer": "..." }}
                ]
            """

            response = my_llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()

           
            try:
                qa_pairs = json.loads(raw)
            except:
                match = re.search(r"\[\s*\{.*?\}\s*\]", raw, re.DOTALL)
                if not match:
                    print("No valid JSON found, skipping chunk.")
                    print("Response:", raw[:200], "...")
                    continue

                json_text = match.group(0)
                qa_pairs = json.loads(json_text)

            # Store results
            for item in qa_pairs:
                examples.append({
                    "question": item["question"],
                    "expected_answer": item["answer"],
                    "relevant_contexts": [chunk]
                })

        return examples

    deepeval_dataset = generate_deepeval_examples(selected_chunks)

    # Save file
    with open(EVAL_OUTPUT_PATH, "w") as f:
        json.dump(deepeval_dataset, f, indent=2)

    print("Dataset created:", len(deepeval_dataset), "examples")


if __name__ == "__main__":

    debug=True
    delete=True

    create_index()
    upload_docs()
    create_eval_dataset(35)
