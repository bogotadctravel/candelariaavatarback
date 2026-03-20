import logging

from dotenv import load_dotenv
from livekit import agents, rtc
from langchain_core.runnables import RunnableConfig
from livekit.agents import (
    cli,
    inference,
    room_io,
    Agent,
    AgentSession,
    AgentServer,
    JobContext,
    JobProcess,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    RoomInputOptions,
    WorkerOptions,
    MetricsCollectedEvent,
    get_job_context
)
from livekit.plugins import deepgram, elevenlabs, langchain, silero, noise_cancellation, openai, hedra
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from PIL import Image
import os
import json
from pprint import pprint
from modules.mainAgent import create_workflow, get_prompt

logger = logging.getLogger("agent")

load_dotenv(".env")

server = AgentServer()

# El VAD se carga de forma lazy (cuando se necesita) para no bloquear el inicio
# Esto evita que el avatar tarde en iniciar mientras carga el modelo

@server.rtc_session(agent_name="avatar-idt-candelaria")
async def my_agent(ctx: JobContext):
    # Logging setup
    #pprint(ctx.__dict__)
    user_id = ctx.room.name
    user_name = ""
    user_phone = ""
    #print(f"Metadata: {ctx}")
    if ctx.job.metadata != '' :
      print(f"Metadata: {ctx.job.metadata}")
      metadata = json.loads(ctx.job.metadata)
      user_id = metadata["user_id"]
      user_name = metadata["user_name"]
      user_phone = metadata["user_phone"]
    else:
      print(f"No metadata")

    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    
    print(f"User id:{user_id}")

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    # El VAD se carga de forma lazy (cuando se necesita)
    session = AgentSession(
      vad=silero.VAD.load(),
      #stt="cartesia/ink-whisper:es",
      #stt=deepgram.STT(model="nova-3", language="es"),
      #tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
      #stt="assemblyai/universal-streaming:en",
      stt=openai.STT(
        model="gpt-4o-transcribe",
      ),
      tts=openai.TTS(
        model="gpt-4o-mini-tts",
        voice="coral",
        instructions="Habla en un tono amigable. Cada vez que te pregunte algo y necesites consultar la respuesta dile que te de un momento para encontrar la mejor respuesta.",
      ),
      turn_detection=MultilingualModel(),
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    avatar_dir = os.path.dirname(os.path.abspath(__file__))
    avatar_image = None
    for ext in ['.png', '.jpg', '.jpeg']:
        image_path = os.path.join(avatar_dir, f'avatar{ext}')
        if os.path.exists(image_path):
            avatar_image = Image.open(image_path)
            break
    
    if not avatar_image:
        raise FileNotFoundError("No avatar image found. Please place an avatar.png, avatar.jpg, or avatar.jpeg in the avatars directory.")
    
    job_context = get_job_context()
    avatar_identity = "static-avatar"
    avatar_session = hedra.AvatarSession(
        avatar_participant_identity=avatar_identity,
        avatar_image=avatar_image,
    )

    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await avatar_session.start(
        session, room=job_context.room
    )


    await session.start(
        room=ctx.room,
        agent=Agent(instructions=get_prompt(),llm=langchain.LLMAdapter(graph= create_workflow(user_id,user_name))),     
        room_input_options=RoomInputOptions(
          noise_cancellation=noise_cancellation.BVC(),
        ),   
    )

    # Join the room and connect to the user
    await ctx.connect()
       
    await session.generate_reply(
        instructions=f"Saluda al usuario y luego preséntate y ofrece tu ayuda. Cada vez que te pregunte algo y necesites consultar la respuesta dile que te de un momento para encontrar la mejor respuesta."
    )


if __name__ == "__main__":
    cli.run_app(server)
