from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import Optional
from pydantic import BaseModel
from services.chatbot_service import ChatbotService
from dependencies import get_current_user

router = APIRouter(prefix="/chatbot", tags=["chatbot"])
chatbot_service = ChatbotService()

class ChatQuery(BaseModel):
    query: str
    reference_pdf: bool = False

@router.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
    success = await chatbot_service.process_pdf(file, str(current_user['_id']))
    if not success:
        raise HTTPException(status_code=500, detail="Failed to process PDF")
        
    return {"message": "PDF processed successfully"}

@router.post("/chat")
async def chat(
    query: ChatQuery,
    current_user: dict = Depends(get_current_user)
):
    response = await chatbot_service.get_chatbot_response(
        query.query,
        str(current_user['_id']),
        query.reference_pdf
    )
    return {"response": response} 