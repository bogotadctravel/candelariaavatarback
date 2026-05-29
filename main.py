from fastapi import FastAPI, Body, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime, timezone
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


def _load_checkpoint_value(r: redis.Redis, key: str) -> dict:
    key_type = r.type(key)

    if key_type == "ReJSON-RL":
        raw_data = r.json().get(key)
    else:
        raw_data = r.get(key)

    if not raw_data:
        return {}

    if isinstance(raw_data, dict):
        return raw_data

    if isinstance(raw_data, str):
        try:
            return json.loads(raw_data)
        except json.JSONDecodeError:
            return {}

    return {}


def _find_nested_string(data, field_name: str) -> str:
    if isinstance(data, dict):
        value = data.get(field_name)
        if isinstance(value, str) and value:
            return value

        for item in data.values():
            found = _find_nested_string(item, field_name)
            if found:
                return found

    if isinstance(data, list):
        for item in data:
            found = _find_nested_string(item, field_name)
            if found:
                return found

    return ""


def _extract_thread_id(checkpoint: dict) -> str:
    configurable = checkpoint.get("configurable", {})
    if isinstance(configurable, dict) and configurable.get("thread_id"):
        return configurable.get("thread_id")

    nested_checkpoint = checkpoint.get("checkpoint", {})
    if isinstance(nested_checkpoint, dict):
        nested_configurable = nested_checkpoint.get("configurable", {})
        if isinstance(nested_configurable, dict) and nested_configurable.get("thread_id"):
            return nested_configurable.get("thread_id")

    return ""


def _extract_client_source(checkpoint: dict) -> str:
    for field in ("client_source",):
        found = _find_nested_string(checkpoint, field)
        if found:
            return found

    return "unknown"


def _extract_checkpoint_ts(checkpoint: dict) -> str:
    ts = checkpoint.get("ts")
    if isinstance(ts, str) and ts:
        return ts

    nested_checkpoint = checkpoint.get("checkpoint", {})
    if isinstance(nested_checkpoint, dict):
        nested_ts = nested_checkpoint.get("ts")
        if isinstance(nested_ts, str) and nested_ts:
            return nested_ts

    return "unknown"


def _parse_datetime_filter(value: Optional[str], end_of_day: bool = False) -> Optional[datetime]:
    if not value:
        return None

    try:
        if len(value) == 10 and value.count("-") == 2:
            parsed = datetime.strptime(value, "%Y-%m-%d")
            if end_of_day:
                return parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
            return parsed

        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except ValueError:
        return None


def _parse_checkpoint_ts(value: str) -> Optional[datetime]:
    if not value or value == "unknown":
        return None

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except ValueError:
        return None


def _extract_messages(checkpoint: dict) -> list:
    channel_values = checkpoint.get("channel_values", {})
    if isinstance(channel_values, dict) and isinstance(channel_values.get("messages"), list):
        return channel_values.get("messages", [])

    nested_checkpoint = checkpoint.get("checkpoint", {})
    if isinstance(nested_checkpoint, dict):
        nested_channel_values = nested_checkpoint.get("channel_values", {})
        if isinstance(nested_channel_values, dict) and isinstance(nested_channel_values.get("messages"), list):
            return nested_channel_values.get("messages", [])

    return []


def _checkpoint_sort_key(item):
    checkpoint, key = item
    ts = checkpoint.get("ts") or checkpoint.get("checkpoint", {}).get("ts") or ""
    checkpoint_id = checkpoint.get("id") or checkpoint.get("checkpoint", {}).get("id") or ""
    return (ts, checkpoint_id, str(key))

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
    respuesta = chat_with_agent(user_id, prompt)
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
        
        # Buscar checkpoints de LangGraph y filtrar por thread_id
        keys = r.keys("langgraph:checkpoints:*")
        if not keys:
            keys = r.keys("checkpoint:*:__empty__:*")
        
        if not keys:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": f"No se encontró conversación para el usuario: {user_id}",
                    "conversation": []
                }
            )
        
        checkpoints = []
        for key in keys:
            checkpoint = _load_checkpoint_value(r, key)
            if not checkpoint:
                continue

            checkpoint_thread_id = _extract_thread_id(checkpoint)
            if checkpoint_thread_id == user_id or f":{user_id}:" in str(key):
                checkpoints.append((checkpoint, key))

        if not checkpoints:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": f"No se encontró conversación para el usuario: {user_id}",
                    "conversation": []
                }
            )

        checkpoint, key = max(checkpoints, key=_checkpoint_sort_key)
        messages = _extract_messages(checkpoint)
        client_source = _extract_client_source(checkpoint)
        conversation_ts = _extract_checkpoint_ts(checkpoint)
        
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
            "client_source": client_source,
            "conversation_ts": conversation_ts,
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
async def list_conversations(
    fecha_desde: Optional[str] = Body(default=None, embed=True),
    fecha_hasta: Optional[str] = Body(default=None, embed=True),
    client_source_filter: Optional[str] = Body(default=None, embed=True),
    current_user: User = Depends(get_current_user)
):
    """
    Lista todas las conversaciones almacenadas con user_id y client_source.
    
    Returns:
        Lista de conversaciones con user_id y client_source
    """
    try:
        redis_url = os.getenv("URL_REDIS")
        r = redis.Redis.from_url(redis_url, decode_responses=True)
        
        # Buscar checkpoints de LangGraph y mantener fallback al formato antiguo
        keys = r.keys("langgraph:checkpoints:*")
        if not keys:
            keys = r.keys("checkpoint:*:__empty__:*")

        conversations_by_pair = {}
        for key in keys:
            # Convertir bytes a string si es necesario
            if isinstance(key, bytes):
                key = key.decode('utf-8')

            checkpoint = _load_checkpoint_value(r, key)
            if not checkpoint:
                continue

            user_id = _extract_thread_id(checkpoint)
            if not user_id and ":" in key:
                parts = key.split(':')
                if len(parts) >= 2:
                    user_id = parts[1]

            if not user_id:
                continue

            client_source = _extract_client_source(checkpoint)
            conversation_ts = _extract_checkpoint_ts(checkpoint)
            pair_key = f"{user_id}:{client_source}"
            current_item = conversations_by_pair.get(pair_key)
            candidate = {
                "user_id": user_id,
                "client_source": client_source,
                "conversation_ts": conversation_ts,
                "key": key,
                "checkpoint": checkpoint,
            }

            if not current_item:
                conversations_by_pair[pair_key] = candidate
                continue

            if _checkpoint_sort_key((checkpoint, key)) > _checkpoint_sort_key((current_item.get("checkpoint", {}), current_item["key"])):
                conversations_by_pair[pair_key] = candidate

        conversations = [
            {
                "user_id": item["user_id"],
                "client_source": item["client_source"],
                "conversation_ts": item["conversation_ts"],
            }
            for item in sorted(
                conversations_by_pair.values(),
                key=lambda item: (item["user_id"], item["client_source"])
            )
        ]

        desde_dt = _parse_datetime_filter(fecha_desde)
        hasta_dt = _parse_datetime_filter(fecha_hasta, end_of_day=True)

        if fecha_desde and not desde_dt:
            raise HTTPException(status_code=400, detail="fecha_desde debe ser una fecha válida ISO o YYYY-MM-DD")

        if fecha_hasta and not hasta_dt:
            raise HTTPException(status_code=400, detail="fecha_hasta debe ser una fecha válida ISO o YYYY-MM-DD")

        if client_source_filter:
            conversations = [item for item in conversations if item["client_source"] == client_source_filter]

        if desde_dt or hasta_dt:
            filtered_conversations = []
            for item in conversations:
                ts_dt = _parse_checkpoint_ts(item["conversation_ts"])
                if not ts_dt:
                    continue

                if desde_dt and ts_dt < desde_dt:
                    continue

                if hasta_dt and ts_dt > hasta_dt:
                    continue

                filtered_conversations.append(item)

            conversations = filtered_conversations
        
        return {
            "status": "success",
            "fecha_desde": fecha_desde,
            "fecha_hasta": fecha_hasta,
            "client_source_filter": client_source_filter,
            "total_conversations": len(conversations),
            "conversations": conversations,
        }
        
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error al listar conversaciones: {str(e)}",
                "traceback": traceback.format_exc(),
                "conversations": []
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
