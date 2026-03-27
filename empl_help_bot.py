
###global variables





SEARCH_MODE = "semantic"

CHAT = "gpt-4o-mini"
emb_model = "text-embedding-3-small"
VECTOR_DIMENSIONS = 1536
top_k = 5

MAX_DEPTH = 1  #for recursive search

debug = False
debug_ready = False
evals=False



### imports

from azure.search.documents import SearchClient
from openai import AzureOpenAI
from pathlib import Path
from azure.core.credentials import AzureKeyCredential
from langchain_core.messages import BaseMessage, HumanMessage
from rich import print
from rich.prompt import Prompt
from init_process import create_index,upload_docs
from evaluation_process import evaluation
from rag_agent_process import rag_agent,test_return
import os
from dotenv import load_dotenv

load_dotenv()

search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_index = os.getenv("AZURE_SEARCH_INDEX")
search_api_key = os.getenv("AZURE_SEARCH_KEY")

service_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
service_api_key = os.getenv("AZURE_OPENAI_KEY")

if not all([search_endpoint, search_index, search_api_key, service_endpoint, service_api_key]):
    raise ValueError("Missing environment variables. Check your .env file.")

search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=search_index,
    credential=AzureKeyCredential(search_api_key),
)

openai_client = AzureOpenAI(
    azure_endpoint=service_endpoint,
    api_key=service_api_key,
    api_version="2024-10-01-preview",
)


def rag_init():
    create_index()
    upload_docs()


### exe code
def main():

    
    conversation_messages: list[BaseMessage] = []
    
    print("[bold green]Welcome to Spyrou Bot![/bold green]\n")

    if debug_ready:  # test prompts from a file
        prompts_file = Path("docs/prompts.txt")
        if prompts_file.exists():
            with prompts_file.open("r") as f:
                file_prompts = [line.strip() for line in f if line.strip()]
            for prompt in file_prompts:
                print(f"[bold blue]You (from file):[/bold blue] {prompt}")
                conversation_messages.append(HumanMessage(content=prompt))
                result = rag_agent().invoke({"messages": conversation_messages})
                ai_message = result["messages"][-1]
                conversation_messages.append(ai_message)
                print(f"[bold green]Spyrou Bot:[/bold green] {ai_message.content}\n")

    while True:  # user inputs
        u_input = Prompt.ask("[bold blue]You[/bold blue]").strip()
        if not u_input:
            continue
        if u_input.lower() == "exit":
            print("[bold red]Goodbye![/bold red]")
            break

        conversation_messages.append(HumanMessage(content=u_input))
        result = rag_agent().invoke({"messages": conversation_messages})
        ai_message = result["messages"][-1]
        conversation_messages.append(ai_message)
        print(f"[bold green]Spyrou Bot:[/bold green] {ai_message.content}\n")





if __name__ == "__main__" and not evals:
    rag_init()
    if debug:
        test_return()
    main()



if __name__ == "__main__" and evals:
    rag_init()
    evaluation()
 
