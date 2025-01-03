from typing import Optional
from fastapi import UploadFile
import PyPDF2
from io import BytesIO
from deep_translator import GoogleTranslator
import google.generativeai as genai
from app.services.db_service import DatabaseService
import os
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotService:
    def __init__(self):
        try:
            # Configure Gemini with experimental flash model
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
                
            genai.configure(api_key=api_key)
            
            # Use Gemini 2.0 Flash experimental model
            generation_config = {
                "temperature": 0.7,
                "max_output_tokens": 2048,
                "top_p": 0.9,
                "top_k": 40,
                "candidate_count": 1,
            }
            
            # Initialize model
            self.model = genai.GenerativeModel(
                model_name='gemini-pro',  # Changed to stable version
                generation_config=generation_config
            )
            
            self.chat_history = []
            self.translator = GoogleTranslator(source='auto', target='bn')
            self.db_service = None
            self.current_conversation = None
            logger.info("ChatbotService initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ChatbotService: {e}")
            raise

    async def initialize_db(self):
        """Initialize database connection"""
        if self.db_service is None:
            try:
                self.db_service = DatabaseService()
                await self.db_service.connect()
            except Exception as e:
                print(f"Database Connection Error: {e}")
                self.db_service = None

    async def cleanup(self):
        """Cleanup database connections"""
        if self.db_service:
            await self.db_service.disconnect()

    async def start_conversation(self, user_id: str = "default_user"):
        await self.initialize_db()
        if self.db_service:
            conversation_id = await self.db_service.create_conversation(user_id)
            self.current_conversation = conversation_id
            return conversation_id
        return None
        
    async def detect_language(self, text: str) -> str:
        """Detect if the input is Banglish or Bangla"""
        try:
            # Simple detection based on character set
            bangla_chars = set('অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়')
            text_chars = set(text)
            if any(char in bangla_chars for char in text_chars):
                return 'bn'
            return 'en'
        except Exception as e:
            print(f"Language detection error: {e}")
            return 'en'
        
    async def convert_banglish_to_bangla(self, text: str) -> str:
        """Convert Banglish text to Bangla"""
        try:
            translation = self.translator.translate(text)
            return translation
        except Exception as e:
            print(f"Translation error: {e}")
            return text
            
    async def process_pdf(self, file: UploadFile, user_id: str):
        """Process and store PDF content"""
        try:
            pdf_content = ""
            pdf_reader = PyPDF2.PdfReader(BytesIO(await file.read()))
            
            for page in pdf_reader.pages:
                pdf_content += page.extract_text()
            
            if not self.current_conversation:
                self.current_conversation = await self.start_conversation(user_id)
                
            await self.db_service.update_pdf_context(self.current_conversation, pdf_content)
            return True
        except Exception as e:
            print(f"PDF processing error: {e}")
            return False
            
    async def get_chatbot_response(self, 
                                 query: str, 
                                 user_id: str, 
                                 reference_pdf: bool = False) -> str:
        """Get response from chatbot"""
        try:
            logger.info(f"Processing query: {query}")
            
            # Detect language and convert if necessary
            lang = await self.detect_language(query)
            logger.info(f"Detected language: {lang}")
            
            if lang != 'bn':
                query = await self.convert_banglish_to_bangla(query)
                logger.info(f"Converted query to Bangla: {query}")

            # Prepare prompt
            system_prompt = """
            You are a helpful assistant that responds in Bangla.
            Provide clear and concise responses.
            If you don't know something, say so in Bangla.
            """

            # Create complete prompt
            full_prompt = f"{system_prompt}\n\nUser: {query}"
            logger.info("Sending request to Gemini")

            try:
                # Get response from Gemini
                chat = self.model.start_chat(history=self.chat_history)
                response = chat.send_message(full_prompt)
                
                if not response or not response.text:
                    logger.error("Empty response from Gemini")
                    return "দুঃখিত, কোনো উত্তর পাওয়া যায়নি। আবার চেষ্টা করুন।"
                
                bot_response = response.text
                logger.info(f"Received response: {bot_response}")

                # Update chat history
                self.chat_history.extend([query, bot_response])
                if len(self.chat_history) > 8:
                    self.chat_history = self.chat_history[-8:]

                return bot_response

            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                return "দুঃখিত, একটি ত্রুটি ঘটেছে। আবার চেষ্টা করুন।"

        except Exception as e:
            logger.error(f"General error in get_chatbot_response: {e}")
            return "দুঃখিত, একটি ত্রুটি ঘটেছে। আবার চেষ্টা করুন।" 