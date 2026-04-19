# & "C:\Program Files\Python311\python.exe" rag_UI.py

import gradio as gr
import requests


API_URL = "http://127.0.0.1:8000/ask"
ASK_URL = "http://127.0.0.1:8000/ask"
LOGIN_URL = "http://127.0.0.1:8000/login"
SIGNUP_URL = "http://127.0.0.1:8000/signup"


def login_user(username, password):
    if not username or not password:
        return None, "Enter username and password"

    r = requests.post(LOGIN_URL, json={
        "username": username,
        "password": password
    })

    if r.status_code == 200:
        return r.json()["user_id"], "Logged in"
    return None, "Invalid credentials"


def signup_user(username, password):
    if not username or not password:
        return "Enter username and password"

    r = requests.post(SIGNUP_URL, json={
        "username": username,
        "password": password
    })

    if r.status_code == 200:
        return "Account created. Now log in."
    return "Signup failed, try a different name"

def ask_hr(message, history, user_id):
    """Send user message to backend and append response to history."""
    if user_id is None:
        return history, user_id, "Please log in first"

    payload = {"query": message, "user_id": user_id}

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            answer = response.json().get("answer", "")
            # Load full conversation from backend if returned
            history_db = response.json().get("history", [])
        else:
            answer = f"Error: {response.status_code}"
            history_db = history
    except Exception as e:
        answer = f"Error: {str(e)}"
        history_db = history

    # Append new message if backend didn't return full history
    if not history_db:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
        return history, user_id, ""

    return history_db, user_id, ""

with gr.Blocks() as demo:

    state = gr.State([])  # stores conversation history
    user_id_state = gr.State(None)  # stores per-tab user_id

    gr.Markdown("## Login / Sign Up")

    username = gr.Textbox(label="Username")
    password = gr.Textbox(label="Password", type="password")

    with gr.Row():
        login_btn = gr.Button("Login")
        signup_btn = gr.Button("Sign Up")

    status = gr.Markdown()

    login_btn.click(
        login_user,
        inputs=[username, password],
        outputs=[user_id_state, status]
    )

    signup_btn.click(
        signup_user,
        inputs=[username, password],
        outputs=[status]
    )

    gr.Markdown("# Spyrou HR RAG Assistant")

    chatbot = gr.Chatbot(label="Conversation")
    user_input = gr.Textbox(
        placeholder="Type your question and press Enter...",
        label="Your Question",
        lines=1
    )

    user_input.submit(
        ask_hr,
        inputs=[user_input, state, user_id_state],
        outputs=[chatbot, user_id_state, user_input]
    )

demo.launch(share=False)

    