from langgraph.checkpoint.redis import RedisSaver
import os
from dotenv import load_dotenv
import redis
import json
# Carga las variables del archivo .env al inicio de la aplicación
load_dotenv()
# Accede a las variables de entorno
URL_REDIS = os.getenv("URL_REDIS")
memory = None
with RedisSaver.from_conn_string(URL_REDIS) as cp:
  cp.setup()
  memory = cp

r = redis.Redis.from_url(URL_REDIS)
# ----------------------- MEMORIA -----------------------
def get_memory():
  # Memoria persistente en Redis por usuario   
  return memory

# ---- FUNCIONES INTERNAS ----

def is_corrupt(history: list) -> bool:
    """
    Devuelve True si:
    - El último mensaje es un assistant con tool_calls
    - Y NO existe respuesta tool con el mismo tool_call_id
    """
    if not history:
        return False

    last = history[-1]

    # Último mensaje = assistant con tool_calls → revisar
    if last.get("role") == "assistant" and last.get("tool_calls"):
        tool_call_id = last["tool_calls"][0]["id"]

        # Verificar si existe respuesta 'tool' correspondiente
        for msg in history:
            if msg.get("role") == "tool" and msg.get("tool_call_id") == tool_call_id:
                return False  # está completo

        return True  # corrupto

    return False


def purge_corrupt_checkpoints():
    keys = r.keys("langgraph:checkpoints:*")
    deleted = []
    skipped = []

    for key in keys:
        raw = r.get(key)
        if not raw:
            continue

        try:
            checkpoint = json.loads(raw)
        except:
            skipped.append(key.decode())
            continue

        # LangGraph guarda los mensajes aquí:
        history = checkpoint.get("channel_values", {}).get("messages", [])

        if is_corrupt(history):
            r.delete(key)
            deleted.append(key.decode())
        else:
            skipped.append(key.decode())

    return {
        "total_checked": len(keys),
        "total_deleted": len(deleted),
        "deleted_keys": deleted,
        "skipped_non_corrupt": len(skipped)
    }
