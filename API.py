# & "C:\Program Files\Python311\python.exe" -m uvicorn API:app --host 127.0.0.1 --port 8000 --reload
# http://127.0.0.1:8000/

from fastapi import FastAPI, Depends,HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from db import SessionLocal
from rag_agent_process import rag_agent
from empl_help_bot import rag_init
from sqlalchemy.orm import Session
from db import SessionLocal
from models import User, ConversationHistory
from auth import hash_password, verify_password
from pydantic import BaseModel

rag_init()
app = FastAPI(title="Spyrou HR RAG API")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class SignupRequest(BaseModel):
    username: str
    password: str

@app.post("/signup")
def signup(req: SignupRequest, db: Session = Depends(get_db)):
    if db.query(User).filter_by(username=req.username).first():
        raise HTTPException(status_code=400, detail="User already exists")

    user = User(
        username=req.username,
        password_hash=hash_password(req.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return {"user_id": user.id}
class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(username=req.username).first()

    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {"user_id": user.id}

class QuestionRequest(BaseModel):
    query: str
    user_id: int

class AnswerResponse(BaseModel):
    answer: str
    history: list[dict] = None


@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest, db: Session = Depends(get_db)):

    # Fetch existing conversation history
    conversation = db.query(ConversationHistory).filter_by(user_id=req.user_id).all()
    history = [{"role": c.role, "content": c.content} for c in conversation]

    # If query is empty, just return history (no dummy AI message)
    if req.query.strip() == "":
        return AnswerResponse(answer="", history=history)

    # Save user message
    user_msg = ConversationHistory(user_id=req.user_id, role="user", content=req.query)
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)
    history.append({"role": "user", "content": req.query})

    # Call RAG agent
    result = rag_agent().invoke({"messages": history})
    ai_content = result["messages"][-1].content

    # Save AI response
    ai_msg = ConversationHistory(user_id=req.user_id, role="assistant", content=ai_content)
    db.add(ai_msg)
    db.commit()
    db.refresh(ai_msg)
    history.append({"role": "assistant", "content": ai_content})

    return AnswerResponse(answer=ai_content, history=history)
