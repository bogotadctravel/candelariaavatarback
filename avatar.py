import asyncio
import logging
import re

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
from livekit.plugins import deepgram, elevenlabs, langchain, silero, noise_cancellation, openai, simli, hedra
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from PIL import Image
import os
import json
from pprint import pprint
from modules.mainAgent import create_workflow, get_prompt

logger = logging.getLogger("agent")

load_dotenv(".env")

server = AgentServer(
    num_idle_processes=3,
    load_threshold=0.95,
)

# El VAD se carga de forma lazy (cuando se necesita) para no bloquear el inicio
# Esto evita que el avatar tarde en iniciar mientras carga el modelo
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

class CandelariaAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ack_task = None
        self._ack_handle = None

    async def _delayed_ack(self):
        try:
            await asyncio.sleep(0.5)
            logger.info("Disparando mensaje de espera por delay de respuesta")
            self._ack_handle = self.session.say(
                "Espera un momento, voy a consultar la información.",
                add_to_chat_ctx=False,
                allow_interruptions=False,
            )
        except asyncio.CancelledError:
            logger.info("Mensaje de espera cancelado antes de reproducirse")
            return
        except Exception as e:
            logger.warning(f"No se pudo reproducir ack de espera: {e}")

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        if self._ack_task and not self._ack_task.done():
            logger.info("Cancelando mensaje de espera anterior")
            self._ack_task.cancel()

        self._ack_handle = None
        logger.info("Programando mensaje de espera por si hay delay en la respuesta")
        self._ack_task = asyncio.create_task(self._delayed_ack())

    def transcription_node(self, text, model_settings):
        async def _clean_text():
            async for chunk in text:
                if isinstance(chunk, str):
                    cleaned = self._remove_json_arrays_preserve_spacing(chunk)
                    cleaned = self._sanitize_for_transcription(cleaned)
                    if cleaned:
                        yield cleaned
                else:
                    yield chunk

        return _clean_text()

    def _sanitize_for_transcription(self, text: str) -> str:
        sanitized = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
        sanitized = re.sub(r"<a\b[^>]*>(.*?)</a>", r"\1", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(
            r"\b(m[aá]s informaci[oó]n|ver m[aá]s|leer m[aá]s|click aqu[ií]|haz click aqu[ií]|enlace|link)\b[:\s]*",
            " ",
            sanitized,
            flags=re.IGNORECASE,
        )
        return sanitized

    def _sanitize_for_speech(self, text: str) -> str:
        sanitized = re.sub(r"https?://\S+|www\.\S+", "", text, flags=re.IGNORECASE)
        sanitized = re.sub(r"<a\b[^>]*>(.*?)</a>", r"\1", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(
            r"\b(m[aá]s informaci[oó]n|ver m[aá]s|leer m[aá]s|click aqu[ií]|haz click aqu[ií]|enlace|link)\b[:\s]*",
            "",
            sanitized,
            flags=re.IGNORECASE,
        )
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        return sanitized

    def _remove_json_arrays(self, text: str) -> str:
        cleaned = text

        # Elimina bloques JSON incrustados, pero conserva el texto alrededor.
        # Se aplica en varias pasadas por si hay más de un bloque.
        previous = None
        while previous != cleaned:
            previous = cleaned
            cleaned = re.sub(r"\{[^{}]*\}", " ", cleaned)
            cleaned = re.sub(r"\[[^\[\]]*\]", " ", cleaned)

        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _remove_json_arrays_preserve_spacing(self, text: str) -> str:
        cleaned = text

        previous = None
        while previous != cleaned:
            previous = cleaned
            cleaned = re.sub(r"\{[^{}]*\}", " ", cleaned)
            cleaned = re.sub(r"\[[^\[\]]*\]", " ", cleaned)

        return cleaned

    def tts_node(self, text, model_settings):
        async def _wrapped():
            if self._ack_task and not self._ack_task.done():
                logger.info("Cancelando mensaje de espera porque ya inició la respuesta final")
                self._ack_task.cancel()

            if self._ack_handle and not self._ack_handle.done():
                try:
                    logger.info("Esperando a que termine el mensaje de espera antes de responder")
                    await self._ack_handle.wait_for_playout()
                except Exception as e:
                    logger.warning(f"No se pudo esperar el ack de espera: {e}")

            async def _clean_text_stream(source):
                async for chunk in source:
                    if isinstance(chunk, str):
                        print(f"TTS raw chunk: {chunk}")
                        original_chunk = chunk
                        chunk = self._remove_json_arrays(chunk)
                        if chunk != original_chunk:
                            print(f"TTS without json arrays: {chunk}")
                        sanitized = self._sanitize_for_speech(chunk)
                        if sanitized != chunk:
                            logger.info("Se sanitizo texto antes de TTS para evitar URLs/enlaces")
                        if sanitized:
                            print(f"TTS sanitized chunk: {sanitized}")
                            yield sanitized
                    else:
                        print(f"TTS non-string chunk: {chunk}")
                        yield chunk

            clean_text = _clean_text_stream(text)

            try:
                async for frame in Agent.default.tts_node(self, clean_text, model_settings):
                    yield frame
            finally:
                self._ack_task = None
                self._ack_handle = None

        return _wrapped()

@server.rtc_session(agent_name="avatar-idt-candelaria")
async def my_agent(ctx: JobContext):
    # Logging setup
    #pprint(ctx.__dict__)
    user_id = ctx.room.name
    user_name = ""
    user_phone = ""
    lang = "es"
    
    if ctx.job.metadata != '' :
      metadata = json.loads(ctx.job.metadata)
      user_id = re.sub(r"\D", "", metadata["user_id"])
      user_name = metadata["user_name"]
      user_phone = metadata["user_phone"]
      lang = metadata["lang"]

    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    # El VAD se carga de forma lazy (cuando se necesita)
    session = AgentSession(
      #   vad=silero.VAD.load(),  # Comenta esto si da problemas de memoria
      #stt="cartesia/ink-whisper:es",
      #stt=deepgram.STT(model="nova-3", language="es"),
      #tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
      #stt="assemblyai/universal-streaming:en",
      vad=ctx.proc.userdata["vad"],
      stt=openai.STT(
        model="gpt-4o-transcribe",
      ),
      tts=openai.TTS(
        model="gpt-4o-mini-tts",
        voice="coral",
        instructions="Habla en un tono amigable. Cada vez que te pregunte algo y necesites consultar la respuesta dile que te de un momento para encontrar la mejor respuesta.",
      ),
      #turn_detection=MultilingualModel(),
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
    # avatar_dir = os.path.dirname(os.path.abspath(__file__))
    # avatar_image = None
    #for ext in ['.png', '.jpg', '.jpeg']:
    #    image_path = os.path.join(avatar_dir, f'avatar{ext}')
    #    if os.path.exists(image_path):
    #        avatar_image = Image.open(image_path)
    #        break
    
    #if not avatar_image:
    #    raise FileNotFoundError("No avatar image found. Please place an avatar.png, avatar.jpg, or avatar.jpeg in the avatars directory.")
    
    # avatar_session = hedra.AvatarSession(
    #    avatar_participant_identity=avatar_identity,
    #    avatar_image=avatar_image,
    # )

    job_context = get_job_context()
    avatar_identity = "static-avatar"
    avatar_session = simli.AvatarSession(
      simli_config=simli.SimliConfig(
         api_key=os.getenv("SIMLI_API_KEY"),
         face_id=os.getenv("SIMLI_FACE_ID"),  
         emotion_id=os.getenv("SIMLI_EMOTION_ID"), #ID of the Simli face to use for your avatar. See "Face setup" for details.
         max_idle_time=600, # 600 segundos para el cierre de simly
         max_session_length=1800, 
      ),
   )
    # avatar_session = hedra.AvatarSession(
    #   avatar_participant_identity=avatar_identity,
    #   avatar_id="f46a3b9d-c6a7-4baf-9593-37b80242813c",
    # )

    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Join the room and connect early to reduce perceived startup delay
    await ctx.connect()

    # Start voice pipeline session first
    await session.start(
        room=ctx.room,
        agent=CandelariaAgent(
            instructions=get_prompt(),
            llm=langchain.LLMAdapter(graph=create_workflow(user_id, user_name, lang, client_source="avatarCDT")),
            allow_interruptions=False,
            min_consecutive_speech_delay=0.4,
            min_endpointing_delay=0.5,
            max_endpointing_delay=1.5,
        ),     
        room_input_options=RoomInputOptions(
          noise_cancellation=noise_cancellation.BVC(),
        ),   
    )

    # Start avatar session after room/session are connected
    await avatar_session.start(
        session, room=ctx.room
    )

    await session.generate_reply(
        instructions=f"Saluda al usuario y luego preséntate y ofrece tu ayuda. Cada vez que te pregunte algo y necesites consultar la respuesta dile que te de un momento para encontrar la mejor respuesta."
    )


if __name__ == "__main__":
    cli.run_app(server)
