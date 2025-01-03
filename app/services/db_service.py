from prisma import Prisma
from datetime import datetime
from typing import Optional, List

class DatabaseService:
    def __init__(self):
        self.db = Prisma()
        self._is_connected = False
        
    async def connect(self):
        if not self._is_connected:
            await self.db.connect()
            self._is_connected = True
    
    async def disconnect(self):
        if self._is_connected:
            await self.db.disconnect()
            self._is_connected = False

    async def create_conversation(self, user_id: str = "default_user") -> str:
        await self.connect()
        conversation = await self.db.conversation.create({
            'data': {
                'userId': user_id,
            }
        })
        return str(conversation.id)

    async def add_message(self, conversation_id: str, sender: str, content: str):
        await self.connect()
        await self.db.message.create({
            'data': {
                'content': content,
                'sender': sender,
                'conversationId': conversation_id,
            }
        })
        
        # Update conversation timestamp
        await self.db.conversation.update({
            'where': {'id': conversation_id},
            'data': {'updatedAt': datetime.now()}
        })

    async def get_conversation(self, conversation_id: str):
        await self.connect()
        return await self.db.conversation.find_unique(
            where={'id': conversation_id},
            include={
                'messages': {
                    'orderBy': {'createdAt': 'asc'}
                }
            }
        )

    async def update_pdf_context(self, conversation_id: str, pdf_content: str):
        await self.connect()
        await self.db.conversation.update({
            'where': {'id': conversation_id},
            'data': {
                'pdfContext': pdf_content,
                'updatedAt': datetime.now()
            }
        }) 