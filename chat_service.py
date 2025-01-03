import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging

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
            
        except Exception as e:
            logger.error(f"Error initializing ChatService: {e}")
            raise

    async def get_response(self, message: str) -> str:
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
            
            return {"response": response.text}
            
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return "দুঃখিত, একটি ত্রুটি ঘটেছে। আবার চেষ্টা করুন।"
