from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from bson import ObjectId

class Message(BaseModel):
    sender: str  # 'user' or 'bot'
    content: str
    timestamp: datetime = datetime.now()

class Conversation(BaseModel):
    user_id: str = "default_user"
    messages: List[Message] = []
    pdf_context: Optional[str] = None
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()

    class Config:
        arbitrary_types_allowed = True 