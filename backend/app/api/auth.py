# backend/app/api/auth.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

router = APIRouter(prefix="/api/auth", tags=["Auth"])

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict

@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    # Credenciales de acceso por defecto para el prototipo
    if req.username == "admin" and req.password == "admin123":
        return {
            "access_token": "mock_jwt_token_admin_authenticated",
            "token_type": "bearer",
            "user": {
                "id": 1,
                "username": "admin",
                "role": "Investigador / Médico",
                "email": "admin@dermai.org"
            }
        }
    raise HTTPException(status_code=401, detail="Credenciales inválidas")
