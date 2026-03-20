from fastapi import FastAPI, Body, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from modules.mainAgent import chat_with_agent
from rag.web_rag import vectorizar_urls
from modules.memory import purge_corrupt_checkpoints
import redis
import json
import os
#imports authentication
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm
from configuration.auth import create_access_token
from configuration.schemas import User, Token, UserInDB
from configuration.dependencies import get_current_user
from configuration.users import UserModel, get_password_hash, authenticate_user
from configuration.database import get_db



app = FastAPI()

origins = [
    "http://localhost:8000",  # Frontend URL , 
    "*",  # Allows all origins (not recommended for production)
] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # allow methods
    allow_headers=["*"], # allow headers
)

class ItemsRequest(BaseModel):
    urllist: List[str]

class ConversationRequest(BaseModel):
    user_id: str
    limit: Optional[int] = 50 

#AUTH METHODS  

@app.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    print(user)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/users/createUsers")
def create_user(user: User, db: Session = Depends(get_db),current_user: User = Depends(get_current_user)):
    # Check if the username already exists
    existing_user = db.query(UserModel).filter(UserModel.username == user.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists",
        )
    print(f"password recibido: {user.password}")

    hashed_password = get_password_hash(user.password)
    # Create a new user object
    new_user = UserModel(username=user.username,email=user.email,hashed_password=hashed_password)
    # Add the new user to the database
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "User created successfully", "user_id": new_user.id}



#CHAT METHODS    

@app.post("/chat")
async def chat(user_id: str = Body(...), prompt: str = Body(...)):
    respuesta = chat_with_agent(user_id,prompt)
    return {"respuesta": respuesta}


@app.post("/create_web_rag")
async def create_web_rag(data: ItemsRequest,current_user: User = Depends(get_current_user)):
    respuesta = vectorizar_urls(data.urllist)
    return {"respuesta": respuesta}


@app.get("/cleanup-corrupt-checkpoints")
def cleanup_endpoint(current_user: User = Depends(get_current_user)):
    """
    Recorre todos los checkpoints de LangGraph;
    elimina SOLO los corruptos (tool_calls sin respuesta).
    """
    result = purge_corrupt_checkpoints()
    return {
        "status": "OK",
        "summary": result
    }

#CONVERSATIONS QUERY METHODS  
@app.post("/conversation")
async def get_conversation(request: ConversationRequest,current_user: User = Depends(get_current_user)):
    """
    Obtiene el historial de conversación de un usuario específico.
    
    Args:
        request: ConversationRequest con user_id y limit opcional
    
    Returns:
        Historial de conversación formateado
    """
    try:
        user_id = request.user_id
        limit = request.limit
        
        # Conectar a Redis
        redis_url = os.getenv("URL_REDIS")
        r = redis.Redis.from_url(redis_url, decode_responses=True)
        
        # Buscar todas las claves de checkpoint para este usuario
        # El formato es: checkpoint:{user_id}:__empty__:{uuid}
        pattern = f"checkpoint:{user_id}:__empty__:*"
        keys = r.keys(pattern)
        
        if not keys:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": f"No se encontró conversación para el usuario: {user_id}",
                    "conversation": []
                }
            )
        
        # Tomar la clave más reciente (la última en la lista)
        key = keys[-1]
        
        # Obtener los datos - manejar tipo ReJSON-RL
        key_type = r.type(key)
        
        if key_type == "ReJSON-RL":
            # Para RedisJSON, usar json().get()
            raw_data = r.json().get(key)
            checkpoint = raw_data if raw_data else {}
        else:
            # Para strings normales
            raw_data = r.get(key)
            checkpoint = json.loads(raw_data) if raw_data else {}
        
        # La estructura real es: checkpoint['checkpoint']['channel_values']['messages']
        checkpoint_data = checkpoint.get("checkpoint", {})
        if isinstance(checkpoint_data, dict):
            channel_values = checkpoint_data.get("channel_values", {})
        else:
            channel_values = {}
        messages = channel_values.get("messages", [])
        
        # Formatear los mensajes
        formatted_messages = []
        # Tomar los últimos 'limit' mensajes (si limit es None, tomar todos)
        messages_to_process = messages[-limit:] if limit else messages
        for msg in messages_to_process:
            formatted_msg = {}  # Inicializar para evitar error de variable posiblemente no definida
            
            # Manejar estructura de LangChain (kwargs) o estructura simple
            if isinstance(msg, dict):
                if "kwargs" in msg:
                    # Estructura LangChain: msg['kwargs']['content'], msg['id'] = ['langchain', 'schema', 'messages', 'HumanMessage']
                    kwargs = msg.get("kwargs", {})
                    content = kwargs.get("content", "")
                    
                    # Determinar el rol basado en el ID
                    msg_id = msg.get("id", [])
                    if isinstance(msg_id, list) and len(msg_id) > 0:
                        msg_type = msg_id[-1]  # Ej: 'HumanMessage', 'AIMessage', 'ToolMessage'
                        if "Human" in msg_type:
                            role = "user"
                        elif "AI" in msg_type:
                            role = "assistant"
                        elif "Tool" in msg_type:
                            role = "tool"
                        else:
                            role = "unknown"
                    else:
                        role = "unknown"
                    
                    formatted_msg = {
                        "role": role,
                        "content": content
                    }
                    
                    # Si es assistant con tool_calls
                    if role == "assistant" and "tool_calls" in kwargs:
                        tool_calls = kwargs.get("tool_calls", [])
                        if tool_calls:
                            formatted_msg["tool_calls"] = [
                                {
                                    "id": tc.get("id"),
                                    "name": tc.get("name") or tc.get("function", {}).get("name"),
                                    "arguments": tc.get("args") or tc.get("function", {}).get("arguments")
                                }
                                for tc in tool_calls
                            ]
                    
                    # Si es tool message
                    if role == "tool":
                        formatted_msg["tool_call_id"] = kwargs.get("tool_call_id")
                        formatted_msg["name"] = kwargs.get("name")
                
                else:
                    # Estructura simple (formato antiguo)
                    formatted_msg = {
                        "role": msg.get("role", "unknown"),
                        "content": msg.get("content", "")
                    }
                    
                    # Tool calls en formato simple
                    if msg.get("role") == "assistant" and "tool_calls" in msg:
                        formatted_msg["tool_calls"] = msg.get("tool_calls", [])
                    
                    if msg.get("role") == "tool":
                        formatted_msg["tool_call_id"] = msg.get("tool_call_id")
                        formatted_msg["name"] = msg.get("name")
            
            formatted_messages.append(formatted_msg)
        
        return {
            "status": "success",
            "user_id": user_id,
            "total_messages": len(messages),
            "returned_messages": len(formatted_messages),
            "conversation": formatted_messages
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error al obtener la conversación: {str(e)}",
                "conversation": []
            }
        )


@app.post("/conversations/list")
async def list_conversations(current_user: User = Depends(get_current_user)):
    """
    Lista todos los usuarios que tienen conversaciones almacenadas.
    
    Returns:
        Lista de user_ids con conversaciones
    """
    try:
        redis_url = os.getenv("URL_REDIS")
        r = redis.Redis.from_url(redis_url, decode_responses=True)
        
        # Buscar todas las claves de checkpoints
        # El formato es: checkpoint:{user_id}:__empty__:{uuid}
        keys = r.keys("checkpoint:*:__empty__:*")
        
        # Extraer los user_ids únicos
        user_ids = set()
        for key in keys:
            # Convertir bytes a string si es necesario
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            
            # Extraer el user_id del formato: checkpoint:{user_id}:__empty__:{uuid}
            parts = key.split(':')
            if len(parts) >= 2:
                user_id = parts[1]
                user_ids.add(user_id)
        
        return {
            "status": "success",
            "total_conversations": len(user_ids),
            "users": sorted(list(user_ids))
        }
        
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error al listar conversaciones: {str(e)}",
                "traceback": traceback.format_exc(),
                "users": []
            }
        )


@app.post("/redis/debug")
async def redis_debug(current_user: User = Depends(get_current_user)):
    """
    Endpoint de debug para ver todas las claves en Redis.
    Útil para diagnosticar problemas de almacenamiento.
    """
    try:
        redis_url = os.getenv("URL_REDIS")
        r = redis.Redis.from_url(redis_url, decode_responses=True)
        
        # Obtener todas las claves
        all_keys = r.keys("*")
        
        # Analizar tipos de claves
        key_analysis = []
        for key in all_keys[:20]:  # Limitar a 20 claves
            key_type = r.type(key)
            key_info = {
                "key": key,
                "type": key_type
            }
            
            # Solo intentar obtener valor si es string
            if key_type == "string":
                try:
                    value = r.get(key)
                    key_info["has_value"] = bool(value)
                    key_info["value_length"] = len(value) if value else 0
                except:
                    key_info["error"] = "Could not get value"
            
            key_analysis.append(key_info)
        
        # Filtrar por tipo de clave
        checkpoint_keys = [k for k in all_keys if "checkpoint" in str(k).lower()]
        langgraph_keys = [k for k in all_keys if "langgraph" in str(k).lower()]
        
        # Buscar claves de checkpoints específicamente
        checkpoint_pattern_keys = r.keys("langgraph:checkpoints:*")
        
        return {
            "status": "success",
            "total_keys": len(all_keys),
            "checkpoint_keys_found": len(checkpoint_keys),
            "langgraph_keys_found": len(langgraph_keys),
            "checkpoint_pattern_match": len(checkpoint_pattern_keys),
            "sample_keys": key_analysis[:10],
            "all_keys_sample": [str(k) for k in all_keys[:20]],
            "redis_url_configured": bool(redis_url),
            "patterns_checked": [
                "*",
                "langgraph:checkpoints:*"
            ]
        }
        
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            }
        )

