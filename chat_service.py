import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging
import uuid
from embedding_service import EmbeddingService
from banglish_service import BanglishService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ChatService:
    def __init__(self):
        try:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found")
            
            genai.configure(api_key=api_key)
            
            # Configure model with simple settings
            generation_config = {
                "temperature": 0.7,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 2048,
            }
            
            self.model = genai.GenerativeModel(
                model_name='gemini-2.0-flash-exp',
                generation_config=generation_config
            )
            
            self.chat = self.model.start_chat(history=[])
            logger.info("ChatService initialized successfully")
            
            self.embedding_service = EmbeddingService()
            self.banglish_service = BanglishService()
            
        except Exception as e:
            logger.error(f"Error initializing ChatService: {e}")
            raise

    async def get_response(self, message: str) -> dict:
        try:
            # Check for Banglish and get correction suggestions
            banglish_correction = None
            if any(ord(c) < 128 for c in message):  # Check if contains ASCII (likely Banglish)
                banglish_correction = await self.banglish_service.get_correction(message)
            
            # Create embedding for the query
            query_embedding = await self.embedding_service.create_embedding(message)
            
            if query_embedding:
                # Find similar cached conversations
                similar_conversations = await self.embedding_service.find_similar_conversations(
                    query_embedding, threshold=0.85
                )
                
                # If similar conversations found, use the most similar one for context
                if similar_conversations:
                    cache_id, similarity = similar_conversations[0]
                    cached_conv = await self.embedding_service.get_cached_conversation(cache_id)
                    if cached_conv:
                        # Add cached conversation as context
                        context = f"""Previous relevant conversation:
                        {cached_conv['text']}
                        
                        Current question: {message}
                        """
                        response = await self._get_gemini_response(context)
                    else:
                        response = await self._get_gemini_response(message)
                else:
                    response = await self._get_gemini_response(message)
            else:
                response = await self._get_gemini_response(message)
            
            # Create a unique conversation ID
            conversation_id = str(uuid.uuid4())
            
            # Format conversation text properly
            conversation_text = f"User: {message}\nBot: {response}"
            
            # Create embedding for the conversation
            conv_embedding = await self.embedding_service.create_embedding(conversation_text)
            
            # Cache the conversation with its embedding
            if conv_embedding:
                cache_name = await self.embedding_service.cache_conversation(
                    conversation_text,
                    embedding=conv_embedding,
                    display_name=f"Conversation_{conversation_id[:8]}"  # Use shorter ID in display name
                )
            else:
                cache_name = None
            
            return {
                "response": response,
                "conversation_id": conversation_id,
                "cache_name": cache_name,
                "has_embedding": conv_embedding is not None,
                "used_cache": bool(similar_conversations),
                "banglish_correction": banglish_correction
            }
            
        except Exception as e:
            logger.error(f"Error in get_response: {e}")
            return {
                "response": "দুঃখিত, একটি ত্রুটি ঘটেছে। আবার চেষ্টা করুন।",
                "error": str(e)
            }

    async def _get_gemini_response(self, message: str) -> str:
        try:
            # Prepare prompt
            prompt = f"""
            You are a helpful AI assistant who always responds in Bangla language.
            If the user writes in Banglish (Bengali written in English), understand it and respond in proper Bangla.
            If the user writes in Bangla, respond in Bangla.
            
            User message: {message}
            """
            
            # Get response
            response = self.chat.send_message(prompt)
            
            if not response or not response.text:
                logger.error("Empty response received")
                return "দুঃখিত, কোনো উত্তর পাওয়া যায়নি। আবার চেষ্টা করুন।"
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return "দুঃখিত, একটি ত্রুটি ঘটেছে। আবার চেষ্টা করুন।"
