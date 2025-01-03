from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class ChatHistory(BaseModel):
    user_id: str
    message: str
    response: str
    model_name: Optional[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    embedding_id: Optional[str] 