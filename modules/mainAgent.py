
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#print(f"OPENAI_API_KEY {OPENAI_API_KEY}")

# Modelo base (puedes usar gpt-4o, gpt-3.5-turbo, etc.)
llm = ChatOpenAI(api_key=OPENAI_API_KEY,model="gpt-4o-mini", temperature=0.5,max_tokens=330,timeout=25)

fecha_hora_actual = datetime.datetime.now()

tools = [search_in_visitbogota, search_in_events, create_web_rag_tool, persona_amable]


# ----------------------- AGENTE -----------------------
class CustomAgentState(AgentState):
    user_id: str
    client_source: str
    preferences: dict


memory = get_memory()
template = f"""
Eres Candelaria, asistente virtual de turismo para Bogotá.

REGLAS BÁSICAS:
- Al inicio de la primera conversación pregunta nombre y lugar de origen.
- Si ya existe en historial, no lo vuelvas a preguntar.
- Consulta historial antes de preguntar datos personales.
- Idioma: SOLO español o inglés, nunca otro idioma.
- Regla estricta de idioma: usa un único idioma durante toda la conversación.
- Si la conversación inicia en inglés, responde SIEMPRE en inglés hasta que termine la sesión.
- Si la conversación inicia en español, responde SIEMPRE en español hasta que termine la sesión.
- No cambies de idioma automáticamente ni porque el usuario mezcle idiomas en un mensaje.
- Si no se indica idioma inicial, usa español por defecto.
- Cierra respuestas útiles con "Bogotá es tu casa".

REGLAS DE RESPUESTA:
- Sé breve: máximo 2-3 oraciones y 150 palabras.
- Responde solo lo preguntado, sin relleno ni contexto histórico largo.
- Si vas a consultar, abre con un ack corto (ej: "Claro, un momento").
- Omite URLs, enlaces y textos web tipo "más información", "ver más", "click aquí", "link", "enlace".
- Si el usuario pide enlace explícitamente, puedes compartirlo.
- No hables de política/corrupción ni de ciudades distintas a Bogotá.
- No respondas solicitudes de imágenes.
- Recomienda zonas gastronómicas, no restaurantes.
- Si piden humano, ofrece WhatsApp de atención (enlace solo si lo piden).
- Si piden itinerario, pregunta: días en Bogotá + interés principal.

HERRAMIENTAS - USA TODAS LAS NECESARIAS:
- search_in_visitbogota: lugares/atractivos, rutas turísticas o gastronómicas, experiencias, recorridos, actividades, zonas.
- create_web_rag_tool: PITs, oficinas de información turística, horarios y contactos de atención.
- search_in_events: eventos, festivales, conciertos, ferias y actividades con fecha.

MULTITOOL - IMPORTANTE:
- Si la pregunta combina temas, consulta todas las herramientas necesarias y consolida en una sola respuesta.
- Ejemplo: "actividades, lugares y eventos" = search_in_visitbogota + search_in_events.

RESULTADOS DE BÚSQUEDA (más de 8 resultados):
- Consulta todos los registros.
- Si hay más de 5, clasifica primero y pregunta filtro antes de listar.
- Eventos: clasifica por tipo.
- Sitios/rutas/atractivos: clasifica por zona (y opcional precio/horario).
- PITs: clasifica por zona y horario.
- Muestra máximo 4 por respuesta y pregunta si desea ver los siguientes 4.

INSTRUCCIONES TÉCNICAS:
- Etiquetas <a> deben llevar class="thelink".
- No crees URLs que inicien con https://visitbogota.co.
- No incluyas links que no provengan de las herramientas.
- En respuestas normales, omite URLs y etiquetas web.
- Si un tool trae HTML/links, usa solo contenido útil.
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
    result = agente.invoke(
        {
          "messages": [{"role": "user", "content": message}],
          "user_id": user_id,  
          "client_source": "chat_web",
          "preferences": {"theme": "dark"} 
        },
        {
            "configurable": {"thread_id": user_id, "client_source": "chat_web"},
            "metadata": {"user_id": user_id, "client_source": "chat_web"},
            "tags": ["chat_web"],
        },
    )
    
    return result["messages"][-1].content


# ----------------------- LANGGRAPH -----------------------

# Estado del grafo
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    client_source: str

# Crear el grafo
def create_workflow(user_id: str, user_name: str = '', lang: str = 'es', client_source: str = 'avatarCDT' )-> StateGraph:
    # Define the graph
    # Tu nodo usando el agente
    print(f"User_ID>{user_id}")
    print(f"user_name>{user_name}")
    print(f"lang>{lang}")

    def agent_node(state: State):
        print("imprimiendo estado")
        print(state)
        print("imprimiendo end estado")
        last_human = next(
            (msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
            None
        )
        txt_name = ''
        if user_name != '' and user_name != 'anonimo' :
          txt_name = f" Este usuario se llama  {user_name} para que lo tengas en cuenta y no tengas que preguntarle el nombre nuevamente solo de donde viene si aún no lo sabes. "

        if lang == 'en'  :
          lenguaje = 'inglés'
          txt_lang = f"IMPORTANT: For this conversation, speak exclusively in English. Do not switch languages at any point during the conversation."
          query = f"Greet at the beginning of the conversation, introduce yourself to the user, and offer your help. {txt_name} {txt_lang}"
        else:
          lenguaje = 'español'
          txt_lang = f"IMPORTANTE: Para esta conversación habla exclusivamente en {lenguaje}. No cambies de idioma durante toda la conversación."
          query = f"Saluda al inicio de la conversación, preséntate ante el usuario y ofrece tu ayuda. {txt_name} {txt_lang}"
        
        
        if last_human:
            query = f"{last_human.content}\n\n{txt_lang}"
        print(f" segundo: ")
        print(f" Query: {query}")

        state_user_id = state.get("user_id", user_id)
        state_client_source = state.get("client_source", client_source)

        print(f"User_ID>{state_user_id}")
        print(f"user_name>{user_name}")
        print(f"lang>{lang}")
        print(f"client_source>{state_client_source}")
        result = agente.invoke(
            {
              "messages": [{"role": "user", "content": query}],
              "user_id": state_user_id,
              "client_source": state_client_source,
              "preferences": {"theme": "dark"} 
            },
            {
                "configurable": {"thread_id": state_user_id, "client_source": state_client_source},
                "metadata": {"user_id": state_user_id, "client_source": state_client_source},
                "tags": [state_client_source],
            },
        )
        
        return {
            "messages": result["messages"],
            "user_id": state_user_id,
            "client_source": state_client_source,
        }

    print(f" primero: ")
    workflow = StateGraph(State)
    workflow.add_node("agent", agent_node)
    workflow.add_edge(START, "agent")
    return workflow.compile()

def get_prompt():
    return template
