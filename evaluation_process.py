

system_prompt = """
You are an HR assistant AI for a company and you are named Spyrou Bot. Your goal is to answer employee
questions accurately based on the HR FAQ documents retrieved by the semantic search tool.
Use only the information provided in the retrieved documents and do not invent answers. If the information is
not present in the retrieved documents, respond accordingly. Provide step-by-step guidance when explaining processes.
Avoid including unrelated information. Do not use references to your documents. 
"""


import os
from tqdm import tqdm  
from pathlib import Path
import json
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric
)
from deepeval.metrics.answer_relevancy.answer_relevancy import AnswerRelevancyMetric
from deepeval.metrics.faithfulness.faithfulness import FaithfulnessMetric
from deepeval.metrics.contextual_recall.contextual_recall import ContextualRecallMetric
from deepeval.metrics.contextual_precision.contextual_precision import ContextualPrecisionMetric
from deepeval.models import AzureOpenAIModel
from rag_agent_process import run_rag_test
from dotenv import load_dotenv

load_dotenv()

search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_index = os.getenv("AZURE_SEARCH_INDEX")
search_api_key = os.getenv("AZURE_SEARCH_KEY")

service_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
service_api_key = os.getenv("AZURE_OPENAI_KEY")

if not all([search_endpoint, search_index, search_api_key, service_endpoint, service_api_key]):
    raise ValueError("Missing environment variables. Check your .env file.")


BASE_DIR = Path(__file__).resolve().parent
telemetry_dir = BASE_DIR / ".deepeval"
telemetry_dir.mkdir(exist_ok=True)
os.environ["DEEPEVAL_TELEMETRY_PATH"] = str(telemetry_dir / ".deepeval_telemetry.txt")

  
def evaluation():
     

    azure_model = AzureOpenAIModel(
        deployment_name="gpt-4o-mini",
        model="gpt-4o-mini",
        base_url=service_endpoint,
        api_key=service_api_key,
        api_version="2024-10-01-preview",
        timeout=300
    )

    # Load dataset
    BASE_DIR = Path(__file__).resolve().parent
    eval_path = BASE_DIR / "evaluations" / "eval_dataset.json"

    with eval_path.open("r", encoding="utf-8") as f:
        eval_data = json.load(f)

    print(f"Loaded {len(eval_data)} entries from the dataset")

    #  Create test cases 
    test_cases = []
    for item in tqdm(eval_data, desc="Running RAG Evaluation"):
        answer, contexts = run_rag_test(item["question"])

        test_cases.append(
            LLMTestCase(
                input=item["question"],
                actual_output=answer,
                expected_output=item["expected_answer"],
                retrieval_context=contexts
            )
        )

    # Define metrics 
    metrics = [
        AnswerRelevancyMetric(threshold=0.7, model=azure_model),
        FaithfulnessMetric(threshold=0.7, model=azure_model),
        ContextualRecallMetric(threshold=0.7, model=azure_model),
        ContextualPrecisionMetric(threshold=0.7, model=azure_model),
    ]

    #  Run evaluation 
    print("\nEvaluation completed. Calculating metrics...\n")

    results = []
    try:
        results = evaluate(test_cases, metrics=metrics)  #,hyperparameters=HYPERPARAMETERS)
    except Exception as e:
        print(f"Error during evaluation: {e}")
 
       
        for case in test_cases:
            results.append(("ERROR", case.input, str(e)))

