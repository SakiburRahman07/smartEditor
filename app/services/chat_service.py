from ..config.database import chat_collection, chroma_collection
from ..models.chat import ChatHistory
from typing import List, Optional
import openai
from datetime import datetime, timedelta
import numpy as np

class ChatService:
    def __init__(self):
        self.openai_client = openai.OpenAI()

    async def _get_embedding(self, text: str) -> List[float]:
        response = await self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    async def _get_relevant_history(self, user_id: str, current_message: str, limit: int = 5) -> List[str]:
        # Get embedding for current message
        current_embedding = await self._get_embedding(current_message)
        
        # Query ChromaDB for similar conversations
        results = chroma_collection.query(
            query_embeddings=[current_embedding],
            n_results=limit
        )
        
        # Get recent chat history from MongoDB
        recent_history = chat_collection.find({
            "user_id": user_id,
            "timestamp": {"$gte": datetime.utcnow() - timedelta(days=7)}
        }).sort("timestamp", -1).limit(limit)
        
        context = []
        for hist in recent_history:
            context.append(f"User: {hist['message']}\nAssistant: {hist['response']}")
        
        return context

    async def process_text_input(
        self,
        text: str,
        user_id: str,
        model_name: Optional[str] = None
    ) -> dict:
        # Get relevant conversation history
        context = await self._get_relevant_history(user_id, text)
        
        # Prepare conversation context
        messages = []
        if context:
            messages.append({
                "role": "system",
                "content": "Previous conversation context:\n" + "\n".join(context)
            })
        
        messages.append({"role": "user", "content": text})
        
        # Get response from OpenAI
        response = await self.openai_client.chat.completions.create(
            model=model_name or "gpt-3.5-turbo",
            messages=messages
        )
        
        response_text = response.choices[0].message.content
        
        # Store in MongoDB
        chat_history = ChatHistory(
            user_id=user_id,
            message=text,
            response=response_text,
            model_name=model_name
        )
        
        # Get embedding and store in ChromaDB
        embedding = await self._get_embedding(text + " " + response_text)
        embedding_id = f"{user_id}_{datetime.utcnow().timestamp()}"
        
        # Store in ChromaDB
        chroma_collection.add(
            embeddings=[embedding],
            documents=[f"User: {text}\nAssistant: {response_text}"],
            ids=[embedding_id],
            metadatas=[{"user_id": user_id, "timestamp": datetime.utcnow().isoformat()}]
        )
        
        # Update MongoDB document with embedding ID
        chat_history.embedding_id = embedding_id
        chat_collection.insert_one(chat_history.dict())
        
        return {
            "response": response_text,
            "context_used": bool(context)
        }

    # ... (keep existing voice and text-to-speech methods)

chat_service = ChatService() 