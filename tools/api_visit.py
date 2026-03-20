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
    Busca información de atractivos turísticos, ofertas turísticas, rutas turísticas, museos y sitios culturales en Bogotá.
    
    ⚠️ IMPORTANTE: Esta herramienta usa una PALABRA CLAVE específica, NO una frase completa.
    El endpoint es: https://api.visitbogota.co/es/api/v2/candelaria_search/{resource}
    Donde {resource} debe ser UNA SOLA PALABRA CLAVE o TÉRMINO CORTO.
    
    📋 CÓMO EXTRAER LA PALABRA CLAVE DE LA PREGUNTA DEL USUARIO:
    
    ❌ NO hagas esto:
    - "Quiero saber qué museos hay en Bogotá" → ❌ INCORRECTO (es una frase)
    - "Dime información sobre rutas turísticas en la ciudad" → ❌ INCORRECTO (es una frase larga)
    - "cuáles son los mejores lugares para visitar" → ❌ INCORRECTO (muy vago)
    
    ✅ SÍ haz esto:
    - "Quiero saber qué museos hay en Bogotá" → ✅ resource = "museos"
    - "Dime información sobre rutas turísticas" → ✅ resource = "rutas turisticas"
    - "¿Dónde hay bibliotecas?" → ✅ resource = "bibliotecas"
    - "Quiero ir a zonas verdes" → ✅ resource = "parques"
    - "Lugares históricos" → ✅ resource = "sitios historicos"
    
    📚 PALABRAS CLAVE VÁLIDAS (ejemplos):
    
    Categorías generales:
    - "museos" (Museo del Oro, Botero, Nacional, etc.)
    - "bibliotecas" (Luis Ángel Arango, Virgilio Barco, etc.)
    - "parques" (Simón Bolívar, Virgilio Barco, etc.)
    - "iglesias" (Primada, San Francisco, etc.)
    - "mercados" (Paloquemao, San Alejo, etc.)
    
    Rutas y experiencias:
    - "rutas turisticas"
    - "rutas culturales"
    - "rutas gastronomicas"
    - "tours"
    - "experiencias"
    
    Sitios específicos:
    - "monserrate"
    - "candelaria"
    - "chorro quevedo"
    - "plaza bolivar"
    - "cerro monserrate"
    
    Zonas y barrios:
    - "usaquen"
    - "chapinero"
    - "la macarena"
    - "zona g"
    - "zona t"
    
    📝 REGLAS PARA CREAR LA PALABRA CLAVE:
    1. Extrae el sustantivo principal de la pregunta
    2. Usa minúsculas SIEMPRE
    3. Si son varias palabras, únelas con ESPACIOS, no con guiones
    4. NO uses guiones (-), mayúsculas, ni caracteres especiales
    5. Sé específico: "museos" es mejor que "lugares"
    6. Si el usuario menciona algo específico, úsalo directamente
    
    🔍 EJEMPLOS DE EXTRACCIÓN:
    
    Usuario: "¿Qué museos puedo visitar en Bogotá?"
    → resource: "museos"
    
    Usuario: "Quiero conocer rutas turísticas"
    → resource: "rutas turisticas"
    
    Usuario: "Dime sobre bibliotecas importantes"
    → resource: "bibliotecas"
    
    Usuario: "Lugares para hacer compras"
    → resource: "mercados"
    
    Usuario: "Quiero ir a la iglesia más antigua"
    → resource: "iglesias"
    
    Usuario: "Parques bonitos para pasear"
    → resource: "parques"
    
    🚫 EJEMPLOS DE CUÁNDO NO USAR ESTA HERRAMIENTA (usar create_web_rag_tool en su lugar):
    
    ❌ Usuario: "¿Dónde hay Puntos de Información Turística?"
    → NO uses search_in_visitbogota → Usa create_web_rag_tool(query="puntos de informacion turistica")
    
    ❌ Usuario: "Quiero saber los horarios de los PITs"
    → NO uses search_in_visitbogota → Usa create_web_rag_tool(query="horarios PITs")
    
    ❌ Usuario: "Necesito ir a una oficina de información turística"
    → NO uses search_in_visitbogota → Usa create_web_rag_tool(query="oficinas informacion turistica")
    
    ❌ Usuario: "¿Cuál es el teléfono del IDT?"
    → NO uses search_in_visitbogota → Usa create_web_rag_tool(query="canales de comunicacion IDT")
    
    ❌ Usuario: "Dónde me pueden atender presencialmente"
    → NO uses search_in_visitbogota → Usa create_web_rag_tool(query="atencion presencial turistas")
    
    ⚠️ RESTRICCIONES CRÍTICAS - LEER ATENTAMENTE:
    
    🚫🚫🚫 ABSOLUTAMENTE PROHIBIDO - NUNCA USES ESTA HERRAMIENTA PARA: 🚫🚫🚫
    
    1. PUNTOS DE INFORMACIÓN TURÍSTICA (PITs)
    2. Oficinas de atención al turista
    3. Centros de información presencial
    4. Teléfonos de contacto del IDT
    5. Horarios de atención de oficinas turísticas
    6. Ubicación de PITs o puntos de información
    7. Canales de comunicación con la oficina de turismo
    
    ❌❌❌ PARA TODO LO ANTERIOR USA: create_web_rag_tool ❌❌❌
    
    💡 DIFERENCIA CLAVE:
    - ESTA HERRAMIENTA (search_in_visitbogota): Busca LUGARES PARA VISITAR (museos, parques, etc.)
    - create_web_rag_tool: Busca DÓNDE OBTENER AYUDA/INFORMACIÓN (PITs, oficinas, teléfonos)
    
    📝 REGLA SIMPLE:
    Si el usuario quiere SABER DE UN LUGAR (museo, parque, iglesia) → usa search_in_visitbogota
    Si el usuario quiere AYUDA O INFORMACIÓN PRESENCIAL (PIT, oficina, teléfono) → usa create_web_rag_tool
    
    Esta información la traes del sitio oficial de Visit Bogotá donde se encuentra la oferta turística de la ciudad.
    Busca la información con esta herramienta; si no encuentras resultados adecuados, puedes buscar en otra fuente del modelo.
    
    🔄 FLUJO DE USO:
    1. Escucha la pregunta del usuario
    2. Identifica el tema principal (museos, parques, etc.)
    3. Extrae la palabra clave en minúsculas con guiones
    4. Usa esa palabra clave como parámetro 'resource'
    5. Si no hay resultados, intenta una palabra clave similar
    
    📋 FORMATO DE RESPUESTA:
    Una vez obtengas los resultados, preséntalos de manera natural y conversacional:
    - NO uses listas numeradas ni bullets
    - NO uses etiquetas como "Nombre:", "Descripción:", "Ubicación:"
    - Escribe en párrafos fluidos integrando nombre, descripción y ubicación
    - Máximo 4 recomendaciones
    - Menciona la ubicación naturalmente dentro del texto
    - Termina con "Recuerda que Bogotá es tu casa"
    """
    url = f"{VISIT_URL}/es/api/v2/candelaria_search/{resource}"
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
    Si el usuario no te indica las fechas, pregúntale el rango de fechas.

    Parámetros (envía cada uno por separado, NO como objeto):
    - fromdate: fecha inicial en formato YYYY-MM-DD (ej: 2025-03-14)
    - tilldate: fecha final en formato YYYY-MM-DD (ej: 2025-03-21)

    Ejemplo de cómo INVOCAR la herramienta:
    search_in_events(fromdate="2025-03-14", tilldate="2025-03-21")

    NO hagas esto:
    - search_in_events({"fromdate": "2025-03-14", "tilldate": "2025-03-21"})  ❌
    - search_in_events(parametros={"fromdate": "2025-03-14", "tilldate": "2025-03-21"})  ❌

    HAZ esto:
    - search_in_events(fromdate="2025-03-14", tilldate="2025-03-21")  ✅

    El parámetro tilldate debe ser al menos un día mayor que fromdate. 

    REGLAS ESTRICTAS PARA FORMATEAR LA RESPUESTA DE EVENTOS:
      
      ❌ **PROHIBIDO - NUNCA uses este formato:**
      - NUNCA uses listas con bullets (1., 2., 3.)
      - NUNCA uses etiquetas literales como "Fechas:", "Descripción:", "Fecha:", "Lugar:", "Nombre:"
      - NUNCA uses negritas para etiquetas (como **Fechas:** o **Descripción:**)
      - NUNCA digas "Más información"
      
      ✅ **FORMATO CORRECTO - SIEMPRE usa este estilo:**
      
      Ejemplo de respuesta CORRECTA:
      "El Festival de Jazz se realizará el próximo fin de semana, el 15 y 16 de febrero en el Parque El Country. 
      Es un evento imperdible donde podrás disfrutar de artistas nacionales e internacionales en un ambiente único. 
      La cita es a partir de las 7:00 PM en la Calle 63 # 47-15. Para más detalles, visita el enlace oficial."
      
      Ejemplo de respuesta INCORRECTA (NUNCA hagas esto):
      ❌ 1. **Festival de Jazz**
         - **Fechas:** 15 y 16 de febrero
         - **Descripción:** Evento con artistas internacionales
         - **Lugar:** Parque El Country
      
      📋 **Instrucciones específicas:**
      1. Escribe eventos como párrafos fluidos, nunca como listas
      2. Integra nombre, fecha y lugar en la misma oración
      3. Usa frases conectivas: "se realizará", "podrás disfrutar", "la cita es"
      4. Para fechas: "el próximo sábado 15 de febrero", "del 20 al 25 de marzo"
      5. Para horarios: "a las 7:00 PM", "desde las 10:00 AM"
      6. Extrae la ubicación del HTML y menciónala naturalmente
      7. Máximo 4 eventos, separados por párrafos, no por números
      8. Si hay múltiples eventos, usa conectores: "También tenemos...", "Otra opción es..."
      9. Termina con: "Recuerda que Bogotá es tu casa"
    
    **No entres en loops:**
      Ejecuta máximo 2 veces esta herramienta y da una respuesta.
    """
    #diccionario = json.loads(texto)

    #fromdate = diccionario['fromdate']
    #tilldate = diccionario['tilldate']

    #fromdate = parametros.fromdate
    #tilldate = parametros.tilldate

    print(fromdate)
    print(tilldate)

    url = f"{VISIT_URL}/es/api/v2/candelaria_events/?tilldate={tilldate}&fromdate={fromdate}"
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
    return "Christian Moreno Piñeros"
