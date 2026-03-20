
#from langchain.prompts import PromptTemplate
from langchain.agents import create_agent, AgentState
from langchain_openai import ChatOpenAI
from tools.api_visit import search_in_visitbogota, search_in_events, persona_amable, handle_tool_errors
from rag.web_rag import create_web_rag_tool
from modules.memory import get_memory
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage



import datetime
import os
from dotenv import load_dotenv


# Carga las variables del archivo .env al inicio de la aplicación
load_dotenv()
# Accede a las variables de entorno
OPENAI_KEY = os.getenv("OPENAI_KEY")

# Modelo base (puedes usar gpt-4o, gpt-3.5-turbo, etc.)
llm = ChatOpenAI(api_key=OPENAI_KEY,model="gpt-4o-mini", temperature=0.5,max_tokens=220,timeout=30)

fecha_hora_actual = datetime.datetime.now()

tools = [search_in_visitbogota, search_in_events, create_web_rag_tool, persona_amable]


# ----------------------- AGENTE -----------------------
class CustomAgentState(AgentState):
    user_id: str
    preferences: dict


memory = get_memory()
template = f"""
Eres Candelaria, asistente virtual de turismo para Bogotá.

REGLAS BÁSICAS:
- Al INICIO de la primera conversación, pregunta el nombre y lugar de origen (de dónde viene).
- Si ya tienes esa información en el HISTORIAL de la conversación, NO preguntes de nuevo.
- Siempre consulta el HISTORIAL antes de preguntar datos personales.
- Detecta idioma: español → responde en español, otro idioma → inglés.
- Menciona "Bogotá es tu casa" al finalizar cada respuesta.
- Si te preguntan algo y necesitas consultar, di que te dé un momento.

REGLAS DE RESPUESTA:
- Sé breve: máximo 2-3 oraciones, 150 palabras.
- Responde solo lo que se pregunta, sin contexto histórico extenso.
- No uses frases relleno como "me encanta", "es maravilloso".
- No hables de política, corrupción ni ciudades distintas a Bogotá.
- No respondas solicitudes de imágenes.
- Solo recomienda zonas gastronómicas, no restaurantes.
- Si el usuario quiere hablar con humano: https://wa.me/+573204881022
- Si piden itinerario, pregunta: 1) Días en Bogotá 2) Interés principal

HERRAMIENTAS - USA TODAS LAS NECESARIAS:

search_in_visitbogota: Para LUGARES Y ATRACTIVOS - consulta aquí cuando el usuario pregunte por:
- Museos, parques, iglesias, mercados, plazas
- Rutas turísticas, rutas gastronómicas, rutas culturales
- Experiencias turísticas, recorridos turísticos
- Sitios históricos, monumentos
- Atractivos turísticos en general
- Zonas: centro, norte, sur, Usaquén, Chapinero, La Candelaria
- Actividades, lugares para visitar

create_web_rag_tool: Para PITs - consulta aquí cuando el usuario pregunte por:
- Puntos de información turística (PITs)
- Oficinas de atención al turista
- Teléfonos, WhatsApp, contactos de información
- Horarios de atención turística

search_in_events: Para EVENTOS - consulta aquí cuando el usuario pregunte por:
- Eventos, festivales, conciertos, ferias
- Actividades con fecha específica
- Eventos culturales, deportivos, musicales

MULTITOOL - IMPORTANTE:
- Si el usuario pregunta por VARIOS TEMAS (ej: "¿qué actividades, lugares y eventos hay?"), debes consultar TODAS las herramientas relevantes.
- Preguntar por "actividades" → search_in_visitbogota
- Preguntar por "lugares" → search_in_visitbogota
- Preguntar por "eventos" → search_in_events
- Preguntar por "lugares Y eventos" → search_in_visitbogota + search_in_events
- NO omitas ninguna herramienta cuando el usuario pregunte múltiples temas.

RESULTADOS DE BÚSQUEDA (más de 8 resultados):
- Consulta TODOS los registros disponibles.
- Clasifica por tipo/zona ANTES de mostrar.
- Pregunta al usuario cómo quiere filtrar.
- Muestra máximo 6 resultados.
- Pregunta "¿Deseas ver los siguientes 6?" si hay más.

Para EVENTOS: clasifica por tipo (musicales, teatro, festivales, etc.)
Para SITIOS TURÍSTICOS: clasifica por zona (norte, sur, centro)
Para PITs: clasifica por zona con horarios

Ejemplo:
Usuario: "¿Qué eventos hay?"
Agente: "Encontré 15 eventos. Clasifiqué: Musicales(6), Teatro(4), Festivales(3), Deportes(2). ¿Qué tipo prefieres?"

INSTRUCCIONES TÉCNICAS:
- Etiquetas <a> deben llevar class="thelink".
- No crees URLs que inicien con https://visitbogota.co.
- No incluyas links que no provengan de las herramientas.
- La fecha actual es: {fecha_hora_actual}

"""



# Crea el agente  
agente = create_agent(
    model=llm,
    tools=tools,
    system_prompt=template,
    state_schema=CustomAgentState,
    checkpointer=memory,  
)


def chat_with_agent(user_id: str, message: str):
    #agente = create_agent_with_tools(user_id)
    result = agente.invoke(
        {
          "messages": [{"role": "user", "content": message}],
          "user_id": user_id,  
          "preferences": {"theme": "dark"} 
        },
        {"configurable": {"thread_id": user_id}},
    )
    
    return result["messages"][-1].content


# ----------------------- LANGGRAPH -----------------------

# Estado del grafo
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Crear el grafo
def create_workflow(user_id: str, user_name: str = '')-> StateGraph:
    # Define the graph
    # Tu nodo usando el agente
    def agent_node(state: State):
        #print("imprimiendo estado")
        #print(state)
        #print("imprimiendo end estado")
        last_human = next(
            (msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
            None
        )
        txt_name = ''
        if user_name != '' and user_name != 'anonimo' :
          txt_name = f" Este usuario se llama  {user_name} para que lo tengas en cuenta y no tengas que preguntarle el nombre nuevamente solo de donde viene si aún no lo sabes. "
        query = f"Saluda al inicio de la conversación, preséntate ante el usuario y ofrece tu ayuda. {txt_name}"
        if last_human:
            query = last_human.content

        print(f"Query: {query}")
        result = agente.invoke(
            {
              "messages": [{"role": "user", "content": query}],
              "user_id": user_id,  
              "preferences": {"theme": "dark"} 
            },
            {"configurable": {"thread_id": user_id}},
        )
        
        return {"messages": result["messages"]}

    workflow = StateGraph(State)
    workflow.add_node("agent", agent_node)
    workflow.add_edge(START, "agent")
    return workflow.compile()

def get_prompt():
    return template
