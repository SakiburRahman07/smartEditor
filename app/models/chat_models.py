from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from bson import ObjectId

class Message(BaseModel):
    sender: str  # 'user' or 'bot'
    content: str
    timestamp: datetime = datetime.now()

class Conversation(BaseModel):
    id: str
    user_message: str
    bot_response: str
    embedding: Optional[List[float]] = None
    cache_name: Optional[str] = None
    created_at: datetime = datetime.now()

class ConversationInDB(Conversation):
    class Config:
        from_attributes = True 