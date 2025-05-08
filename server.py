
from fastapi import FastAPI, UploadFile, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt
from jwt.exceptions import InvalidTokenError
import os
from datetime import datetime, timedelta

from pipeline import TransferPipeline

import io

app = FastAPI()

# Add CORS middleware BEFORE your routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Your frontend URLs
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Include OPTIONS
    allow_headers=["*"],
)

# Supabase auth. The secret is read from the environment, see the README.
SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://oxxjmxltynpzjnmbvivz.supabase.co")
ALGORITHM = "HS256"

# Security scheme
security = HTTPBearer()

transferred = {}
pipeline = TransferPipeline()

class TokenData(BaseModel):
    user_id: str = ""
    email: str = ""
    role: str = ""

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Supabase JWT token and return user data"""
    if not SUPABASE_JWT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SUPABASE_JWT_SECRET environment variable is not set",
        )
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=[ALGORITHM], audience="authenticated", issuer=f"{SUPABASE_URL}/auth/v1")
        
        # Extract user data from Supabase JWT
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        role: str = payload.get("role")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token - missing user ID",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        return TokenData(user_id=user_id, email=email, role=role)
        
    except InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

def perform_transfer(task_id, content_bytes, style_bytes):
    transferred[task_id] = pipeline.transfer(content_bytes, style_bytes, do_segmentation=False, strength=1.0)

@app.post("/transfer")
async def handle_transfer(contentFile: UploadFile,styleFile: UploadFile, background_tasks: BackgroundTasks, current_user: TokenData = Depends(verify_token)):
    task_id = str(len(background_tasks.tasks) + 1)
    background_tasks.add_task(perform_transfer, task_id, await contentFile.read(), await styleFile.read())
    return {"taskID": task_id}

@app.get("/jobs/{task_id}")
async def get_transferred(task_id: str):
    stylized = transferred.get(task_id)
    if not stylized:
        return {"noo": "NOT DONE!"}
    return StreamingResponse(io.BytesIO(stylized), media_type="image/png")