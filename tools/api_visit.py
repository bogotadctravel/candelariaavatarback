import requests
import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

# Carga las variables del archivo .env al inicio de la aplicación
load_dotenv()
# Accede a las variables de entorno
VISIT_URL = os.getenv("VISIT_URL")


@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )
 

 
@tool
def search_in_visitbogota(resource:str):
    """
    Busca en Visit Bogotá atractivos turísticos, rutas, experiencias, museos, sitios culturales y zonas de Bogotá.

    Usa `resource` como una sola palabra clave o término corto, en minúsculas y sin caracteres especiales. Ejemplos válidos: `museos`, `bibliotecas`, `parques`, `iglesias`, `mercados`, `rutas turisticas`, `rutas culturales`, `rutas gastronomicas`, `tours`, `experiencias`, `monserrate`, `candelaria`, `chorro quevedo`, `plaza bolivar`, `cerro monserrate`, `usaquen`, `chapinero`, `la macarena`, `zona g`, `zona t`.

    Úsala cuando el usuario pregunte por lugares para visitar, rutas, experiencias o zonas. No la uses para PITs, oficinas de atención, teléfonos, horarios de servicio, centros de información o canales institucionales; para eso usa `create_web_rag_tool`.

    Resume la respuesta en párrafos fluidos, sin listas ni etiquetas como "Nombre:", "Descripción:" o "Ubicación:", con máximo 4 recomendaciones y cerrando con "Recuerda que Bogotá es tu casa".
    """
    url = f"{VISIT_URL}/es/api/v3/candelaria_search/{resource}"
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        response = response.json()
        return response
    else:
        data = response.json()
        print(data)
        return False

class EventParams(BaseModel):
    fromdate: str 
    tilldate: str 

@tool
def search_in_events(fromdate: str, tilldate: str):
    """
    Busca eventos próximos en Bogotá.

    Recibe `fromdate` y `tilldate` por separado en formato `YYYY-MM-DD`, por ejemplo: `search_in_events(fromdate="2025-03-14", tilldate="2025-03-21")`. Si el usuario no da fechas, pídelas; `tilldate` debe ser posterior a `fromdate`.

    Úsala cuando el usuario pregunte por eventos, festivales, conciertos, ferias o actividades con fecha.

    Responde en párrafos fluidos, sin listas ni etiquetas como "Fechas:", "Descripción:", "Lugar:" o "Nombre:", con máximo 4 eventos y cerrando con "Recuerda que Bogotá es tu casa".
    """
    #diccionario = json.loads(texto)

    #fromdate = diccionario['fromdate']
    #tilldate = diccionario['tilldate']

    #fromdate = parametros.fromdate
    #tilldate = parametros.tilldate

    print(fromdate)
    print(tilldate)

    url = f"{VISIT_URL}/es/api/v3/candelaria_events/?tilldate={tilldate}&fromdate={fromdate}"
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        response = response.json()
        return response
    else:
        data = response.json()
        print(data)
        return False

@tool
def persona_amable (text: str) -> str:
    '''Retorna la persona más amable. Se espera que la entrada esté vacía "" 
    y retorna la persona más amable del universo'''
    return [{
        "respuesta": "\n\n".join(["Christian Moreno"])
    }]
