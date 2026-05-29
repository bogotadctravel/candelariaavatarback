from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
# Carga las variables del archivo .env al inicio de la aplicación
load_dotenv()
# Accede a las variables de entorno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize your LLM
model = ChatOpenAI(api_key=OPENAI_API_KEY,model="gpt-4o-mini", temperature=0.5,max_tokens=1000,timeout=30)

# Create the agent
agent = create_agent(
    model=model,
    system_prompt="You are a helpful assistant.",
    # You can also pass tools here if needed:
    # tools=[your_tool_list]
)

# Invoke the agent
async def run_agent():
    result = await agent.ainvoke({"messages": [HumanMessage(content="Hello, how are you?")]})
    print(result)

# Call the async function
import asyncio 
asyncio.run(run_agent())
