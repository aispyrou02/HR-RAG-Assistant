# 🤖 HR-RAG-Assistant (Spyrou Bot)

An AI-powered HR assistant that answers employee questions using Retrieval-Augmented Generation (RAG) with Azure OpenAI and Azure Cognitive Search.

This project implements a full pipeline including:

* Document ingestion and semantic chunking
* Vector + hybrid retrieval strategies
* LLM-based reasoning with LangGraph agents
* Evaluation using DeepEval
* Optional API, database, and UI integration

---

## 🚀 Features

* 🔍 **Semantic / Hybrid / Recursive / Hierarchical Retrieval**
* 🧠 **LangGraph Agent with Tool Calling**
* 📄 **PDF ingestion + advanced preprocessing**
* 📊 **Evaluation pipeline (Faithfulness, Relevancy, Recall, Precision)**
* 💾 **User system + conversation history (SQLAlchemy)**
* 🌐 **FastAPI backend**
* 🖥 **Optional Gradio UI**

---

## 📁 Project Structure

```
hr-rag-assistant/
│
├── app/
│   ├── empl_help_bot.py.py                  # CLI entry     (main)
│   ├── rag_agent_process.py
│   ├── init_process.py
│   ├── evaluation_process.py
│   ├── models.py
│   ├── API.py
│   ├── auth.py
│   ├── db.py
|   ├── rag_UI.py
│
├── docs/
│   ├── hr_faqs.pdf
│   ├── prompts.txt
│
├── evaluations/
│   └── eval_dataset.json
│
├── requirements.txt
├── README.md
```

---

## 🔐 Environment Setup

Create a `.env` file in the root directory:

```
AZURE_SEARCH_ENDPOINT=your_endpoint
AZURE_SEARCH_INDEX=your_index
AZURE_SEARCH_KEY=your_key

AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_KEY=your_key

DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/db_name
```

---

## 🧪 Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ 1. Simple Run (CLI RAG Assistant)

This runs the chatbot directly in the terminal.

```bash
python empl_help_bot.py
```

What happens:

* Creates Azure Search index
* Uploads HR documents
* Starts interactive chat

Example:

```
You: How can I apply for parental leave?
Spyrou Bot: ...
```

Type `exit` to quit.

---

# 🧪 2. Run Evaluation

To evaluate your RAG system:

In `empl_help_bot.py` set:

```python
evals = True
```

Then run:

```bash
python empl_help_bot.py
```

This will:

* Run dataset-based evaluation
* Compute metrics (faithfulness, relevancy, etc.)

---

# 🌐 3. Full System (API + DB + UI)

## 🧩 Step 1 — Start FastAPI backend

```bash
python -m uvicorn API:app --reload
```

API will run at:

```
http://127.0.0.1:8000
```

---

## 💾 Step 2 — Database setup

Make sure PostgreSQL is running and your `.env` contains:

```
DATABASE_URL=postgresql+psycopg2://...
```

Tables will be created automatically via SQLAlchemy.

---

## 🖥 Step 3 — Run Gradio UI (if available)

```bash
python rag_UI.py
```

This provides:

* chat interface
* user login/signup
* conversation history

---

# 🧠 Retrieval Modes

You can modify retrieval behavior in code:

* `semantic`
* `hybrid`
* `recursive`
* `hierarchical`

This allows experimentation with:

* recall vs precision
* multi-step retrieval

---

# 📊 Evaluation Metrics

The system uses DeepEval:

* Answer Relevancy
* Faithfulness
* Contextual Recall
* Contextual Precision

---

# ⚠️ Notes

* Requires Azure OpenAI + Azure Search setup
* Large documents may take time to process
* Ensure API keys are valid

---

# 📌 Future Improvements

* Add authentication tokens (JWT)
* Deploy API (Docker / cloud)
* Improve UI experience
* Optimize chunking performance

---

# 👨‍💻 Author

Agamemnon-Ioannis Spyrou


---

