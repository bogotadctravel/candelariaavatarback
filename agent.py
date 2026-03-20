from dotenv import load_dotenv
from livekit import agents, rtc
from langchain_core.runnables import RunnableConfig
from livekit.agents import AgentServer, AgentSession, RoomInputOptions, JobProcess,inference
from livekit.plugins import deepgram, elevenlabs, langchain, silero, noise_cancellation, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import (
    cli,
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    RoomInputOptions,
    WorkerOptions,
    MetricsCollectedEvent
)


#from livekit.plugins.langchain import LLMAdapter

from modules.mainAgent import create_workflow, get_prompt

load_dotenv(".env.local")

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):

    # Puedes usar el ID de la sala como user_id o generar uno por usuario
    user_id = ctx.room.name or "80144832"
    """tts=inference.TTS(
        model="cartesia/sonic-3", 
        voice="5c5ad5e7-1020-476b-8b91-fdcbe9cc313c", 
        language="es",
        extra_kwargs={
            #"speed": 1.5,
            #"volume": 1.2,
            #"emotion": "excited"
        }
      ),
    """
    session = AgentSession(
      vad= silero.VAD.load(),
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
        instructions="Habla en un tono amigable",
      ),
      turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Agent(instructions=get_prompt(),llm=langchain.LLMAdapter(graph= create_workflow())),     
        room_input_options=RoomInputOptions(
          noise_cancellation=noise_cancellation.BVC(),
        ),   
    )
    """
    background_audio = BackgroundAudioPlayer(
        # play office ambience sound looping in the background
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.9),
        # play keyboard typing sound when the agent is thinking
        thinking_sound=[
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.9),
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.9),
        ],
    )

    await background_audio.start(room=ctx.room, agent_session=session)
    """
    await ctx.connect()

    await session.generate_reply(
        instructions="Saluda al usuario y luego preséntate y ofrece tu ayuda"
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
