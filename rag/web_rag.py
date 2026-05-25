
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.tools import tool
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from dotenv import load_dotenv
# Carga las variables del archivo .env al inicio de la aplicación
load_dotenv()
# Accede a las variables de entorno
PATH_VECTOR_DB = os.getenv("PATH_VECTOR_DB")
# Accede a las variables de entorno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def scrape_url(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # extraer texto visible
    for script in soup(["script", "style", "noscript"]):
        script.extract()

    text = soup.get_text(separator="\n")
    cleaned = "\n".join(line.strip() for line in text.split("\n") if line.strip())

    return cleaned

def scrape_to_documents(urls: list[str]):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, add_start_index=True)

    for url in urls:
        try:
            print(f"Scrapeando: {url}")
            content = scrape_url(url)
            doc = Document(page_content=content, metadata={"source": url})
            chunks = splitter.split_documents([doc])
            docs.extend(chunks)
        except Exception as e:
            print(f"Error en {url}: {e}")

    return docs

def vectorizar_urls(urls: list[str]):
    index_path = f"{PATH_VECTOR_DB}/web_index"
    docs = scrape_to_documents(urls)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local(index_path)
    return f"Index web guardado en: {index_path}"
    

# 🔥 TOOL 2 — RAG desde web scraping
@tool
def create_web_rag_tool(query: str)-> str:
    """
    Busca información sobre Puntos de Información Turística (PITs) y canales de comunicación en Bogotá.
    
    📍 ¿QUÉ SON LOS PITS?
    Los Puntos de Información Turística (PITs) son espacios físicos distribuidos por Bogotá donde los 
    turistas y residentes pueden obtener información, mapas, orientación y asistencia sobre la ciudad.
    
    🎯 CUÁNDO USAR ESTA HERRAMIENTA - REGLAS ESTRICTAS:
    
    ✅ USA ESTA HERRAMIENTA EXCLUSIVAMENTE cuando el usuario pregunte por:
    
    1. Puntos de Información Turística (PITs)
       - "¿Dónde hay PITs en Bogotá?"
       - "Puntos de información turística"
       - "PIT" o "PITs"
    
    2. Oficinas de información turística PRESENCIALES
       - "¿Dónde me pueden atender presencialmente?"
       - "Oficinas de turismo"
       - "Centros de atención al turista"
    
    3. Información de CONTACTO del IDT/Oficina de turismo
       - "Teléfono de información turística"
       - "WhatsApp de la oficina de turismo"
       - "Cómo contactar al IDT"
       - "Canales de comunicación"
       - "Línea de atención"
    
    4. Horarios y ubicaciones de ATENCIÓN PRESENCIAL
       - "Horarios de los PITs"
       - "¿A qué hora abren las oficinas de información?"
       - "¿Dónde queda el punto de información más cercano?"
    
    5. Servicios de AYUDA al turista
       - "Necesito ayuda presencial"
       - "Dónde puedo obtener información en persona"
       - "Mapas y folletos en físico"
    
    🚫 NO USES ESTA HERRAMIENTA PARA:
    
    ❌ Buscar atractivos turísticos → Usa search_in_visitbogota
       (museos, parques, iglesias, mercados, rutas, etc.)
    
    ❌ Buscar eventos → Usa search_in_events
       (festivales, conciertos, ferias, etc.)
    
    ❌ Información general de Bogotá → Responde directamente o usa otra fuente
    
    ❌ Restaurantes, hoteles o comercios → Indica que no tienes esa información
    
    💡 DIFERENCIA CLAVE PARA NO CONFUNDIRTE:
    
    🤔 Si el usuario quiere VISITAR UN LUGAR (museo, parque, iglesia):
    → Usa search_in_visitbogota ❌ NO uses esta herramienta
    
    🤔 Si el usuario quiere AYUDA, INFORMACIÓN o CONTACTO:
    → Usa ESTA HERRAMIENTA (create_web_rag_tool) ✅
    
    📋 EJEMPLOS CLAROS:
    
    ✅ Pregunta: "¿Dónde hay PITs?" 
    ✅ Respuesta: Usa create_web_rag_tool(query="puntos de informacion turistica")
    
    ❌ Pregunta: "¿Qué museos hay?"
    ❌ Respuesta INCORRECTA: Usar create_web_rag_tool
    ✅ Respuesta CORRECTA: Usar search_in_visitbogota(resource="museos")
    
    ✅ Pregunta: "¿Cuál es el teléfono de información turística?"
    ✅ Respuesta: Usa create_web_rag_tool(query="telefono informacion turistica")
    
    📋 ESTRUCTURA DE LA RESPUESTA:
    Cuando encuentres información de PITs, presenta de manera natural y concisa:
    
    ✅ FORMATO CORRECTO:
    "Encontré [N] Puntos de Información Turística en Bogotá:
    
    📍 [Nombre del PIT] - [Dirección específica]
    🕒 Horario: [Horarios de atención]
    📞 Contacto: [Teléfono si aplica]
    
    📍 [Nombre del siguiente PIT]..."
    
    🎨 REGLAS DE FORMATO:
    1. Usa emojis (📍, 🕒, 📞) para organizar la información
    2. Máximo 4-5 PITs por respuesta
    3. Incluye siempre: nombre, dirección y horario
    4. Si hay teléfono o email, inclúyelo
    5. NO uses listas con bullets o numeración larga
    6. Presenta la información en párrafos cortos o líneas con emojis
    7. Si pregunta por un PIT específico, da detalles completos de ese único PIT
    
    📞 CANALES DE COMUNICACIÓN:
    Si el usuario pregunta por canales de contacto o comunicación de la red turística,
    busca la información y preséntala claramente con:
    - Teléfono/WhatsApp
    - Email
    - Redes sociales (si aplica)
    - Horario de atención telefónica
    
    🔍 EJEMPLOS DE CONSULTAS VÁLIDAS:
    - "¿Dónde hay PITs en Bogotá?"
    - "Puntos de información turística"
    - "Oficinas de turismo presenciales"
    - "¿Cuál es el teléfono de información turística?"
    - "Horarios de atención de los PITs"
    - "Canales de comunicación turística"
    - "¿Dónde me pueden ayudar presencialmente?"
    
    ⚠️ IMPORTANTE:
    - Esta herramienta busca en una base de datos local de PITs de Bogotá
    - Si no encuentras información relevante, indica amablemente que no tienes datos actualizados
    - Recuerda mencionar: "Recuerda que Bogotá es tu casa" al finalizar
    """

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.load_local(f"{PATH_VECTOR_DB}/web_index", embeddings, allow_dangerous_deserialization=True)

    docs = db.similarity_search(query, k=10)
    return "\n\n".join([d.page_content for d in docs])
