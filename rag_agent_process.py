

SEARCH_MODE = "semantic"

CHAT = "gpt-4o-mini"
emb_model = "text-embedding-3-small"
VECTOR_DIMENSIONS = 1536
top_k = 15

MAX_DEPTH = 1  #for recursive search

debug=False





### imports
from numpy import dot
from numpy.linalg import norm
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from azure.search.documents.models import VectorizedQuery
from langchain_openai import AzureChatOpenAI
from azure.search.documents import SearchClient
from openai import AzureOpenAI
from langgraph.graph import StateGraph, END, START
from azure.core.credentials import AzureKeyCredential
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from operator import add as add_messages
from langchain_core.tools import tool
from rich import print
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



def proxy_metrics(results):
    """
    Calculate retrieval metrics reliably.
    Ensures top_score >= avg_score >= lowest_score.
    """
    if not results:
        return {"top_score": 0.0, "avg_score": 0.0, "score_gap": 0.0}

    # sort results by @search.score descending
    results_sorted = sorted(results, key=lambda d: d.get("@search.score", 0), reverse=True)

    scores = [d.get("@search.score", 0) for d in results_sorted]
    top_score = scores[0]
    avg_score = sum(scores) / len(scores)
    score_gap = top_score - scores[-1]

    return {
        "top_score": top_score,
        "avg_score": avg_score,
        "score_gap": score_gap,
    }

def prune_by_score_gap(docs, gap_ratio=0.85):
    if not docs:
        return docs

    scores = [d.get("@search.score", 0) for d in docs]
    max_score = max(scores)

    return [
        d for d in docs
        if d.get("@search.score", 0) >= max_score * gap_ratio
    ]

def dynamic_top_k_from_scores(docs):
    scores = [d.get("@search.score", 0) for d in docs]
    if not scores:
        return 0

    gap = max(scores) - min(scores)

    if gap > 0.02:
        return 2
    if gap > 0.01:
        return 3
    return 5





def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def _retriever_tool_internal(query: str, depth: int = 0):
    """
    Internal retrieval logic supporting semantic, hybrid, recursive, and hierarchical search.
    """

    query_embedding = openai_client.embeddings.create(
        model=emb_model, input=query
    ).data[0].embedding

   
    vector_queries = [
        VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top_k,
            fields="summary_vector",
            weight=1.0
        )
    ]

    # Determine search_text
    if SEARCH_MODE in ["semantic", "recursive", "hierarchical"]:
        search_text = "*"
    elif SEARCH_MODE == "hybrid":
        search_text = query

    # Perform search
    results = search_client.search(
        search_text=search_text,
        vector_queries=vector_queries,
        select=["content", "summary"]
    )

    top_results = list(results)[:top_k]

    # ---------------- RECURSIVE MODE ----------------
    if SEARCH_MODE == "recursive" and depth < MAX_DEPTH and top_results:
        recursive_results = []
        for doc in top_results:
            sub_chunks = doc["content"].split(". ")[:top_k]
            for chunk in sub_chunks:
                recursive_results.extend(
                    _retriever_tool_internal(chunk, depth=depth + 1)
                )
        top_results.extend(recursive_results)

    # ---------------- HIERARCHICAL MODE ----------------
    if SEARCH_MODE == "hierarchical" and top_results:
        hierarchical_results = []

        top_scores = [doc.get("@search.score", 0) for doc in top_results]
        max_score = max(top_scores) if top_scores else 1.0
        min_score = min(top_scores) if top_scores else 0.0
        range_score = max_score - min_score if max_score != min_score else 1.0

        for doc in top_results:
            sub_chunks = doc["content"].split(". ")[:top_k]

            for chunk in sub_chunks:
                chunk_embedding = openai_client.embeddings.create(
                    model=emb_model, input=chunk
                ).data[0].embedding

                score = cosine_similarity(query_embedding, chunk_embedding)

               
                normalized_score = min_score + ((score + 1) / 2) * range_score

                hierarchical_results.append({
                    "content": chunk,
                    "summary": doc.get("summary", ""),
                    "@search.score": normalized_score
                })

        top_results.extend(hierarchical_results)

    return top_results





@tool
def retriever_tool(query: str):
    """
    This tool searches and returns information about the HR policies
    """
    top_results = _retriever_tool_internal(query, depth=0)
    top_results = prune_by_score_gap(top_results, gap_ratio=0.9)
    #k = min(dynamic_top_k_from_scores(top_results), 3)
    k=5
    top_results = top_results[:k]

    if debug:
        stats = proxy_metrics(top_results)
        print("Retrieval Proxy Metrics")
        for k, v in stats.items():
            print(f"{k}: {v:.3f}")
        print("\n")

    if not top_results:
        return "No relevant information found in the documents"
    final_results = []
    for i, doc in enumerate(top_results):
        final_results.append(
            f"""Document {i+1} Summary:
    {doc.get("summary", "N/A")}

    Content (optional):
    {doc["content"]}
    """
        )

    return "\n\n".join(final_results)




tools = [retriever_tool]


def test_return():
    global SEARCH_MODE
    global debug
    debug=True
    temp=SEARCH_MODE
    queries = [
        "How can employees apply for parental leave?",
        "What to do when i receive my military papers?",
        "How do I use the self-service portal?",
        "What to do when i am entitled to a car",
    ]

    SEARCH_MODE ="semantic"
    print(SEARCH_MODE+"\n")
    for query in queries:
        ds = retriever_tool.run(query)

    
    SEARCH_MODE ="hybrid"
    print(SEARCH_MODE+"\n")
    for query in queries:
        ds = retriever_tool.run(query)


    SEARCH_MODE = "recursive"
    print(SEARCH_MODE+"\n")
    for query in queries:
        ds = retriever_tool.run(query)


    SEARCH_MODE="hierarchical"
    print(SEARCH_MODE+"\n")
    for query in queries:
        ds = retriever_tool.run(query)


    SEARCH_MODE = temp
    debug=False

#########

load_dotenv()

my_llm = AzureChatOpenAI(
    azure_endpoint=service_endpoint,
    api_key=service_api_key,
    api_version="2024-10-01-preview",
    azure_deployment=CHAT,
    temperature=0.0,
)

my_llm = my_llm.bind_tools(tools)


class mystate(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: mystate):
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0


system_prompt = """
You are an HR assistant AI for a company and you are named Spyrou Bot. Your goal is to answer employee
questions accurately.Use only the information provided in the retrieved documents.Do NOT invent information. 
If the information is not present in the retrieved documents, respond accordingly. Provide step-by-step guidance when explaining processes.
Avoid including unrelated information. Do not use references to your documents. 
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools}


def llm_agent(state: mystate) -> mystate:
    messages = state["messages"]
    messages = [SystemMessage(content=system_prompt)] + messages
    message = my_llm.invoke(messages)
    return {"messages": [message]}


def action_agent(state: mystate) -> mystate:
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        if debug:
            print("calling tool " + t["name"])
        if t["name"] not in tools_dict:
            if debug:
                print("no such tool")
            result = "No tool with this name found, try again"
        else:
            result = tools_dict[t["name"]].invoke(t["args"].get("query", ""))
        results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result)))
        if debug:
            print("tool exe complete")
    return {"messages": results}



def rag_agent():



    my_graph = StateGraph(mystate)
    my_graph.add_node("llm", llm_agent)
    my_graph.add_node("retriever", action_agent)
    my_graph.add_conditional_edges("llm", should_continue, {True: "retriever", False: END})
    my_graph.add_edge("retriever", "llm")
    my_graph.add_edge(START, "llm")
    rag_agent = my_graph.compile()
    return rag_agent

##for testing 
def run_rag_test(query: str):

    
    result = rag_agent().invoke({
        "messages": [HumanMessage(content=query)]
    })

    answer = result["messages"][-1].content

    retrieved_texts = retriever_tool.run(query)
    import re
    contexts = re.split(r'Document \d+ Summary:', retrieved_texts)
    # Clean up empty strings and whitespace
    contexts = [c.strip() for c in contexts if c.strip()]

    return answer, contexts



def spyrou_rag_agent(query: str):

    
    result = rag_agent().invoke({
        "messages": [HumanMessage(content=query)]
    })

    answer = result["messages"][-1].content


    return answer