from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import AgentServer,AgentSession, Agent, room_io
from livekit.plugins import noise_cancellation, silero, langchain, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from modules.mainAgent import create_workflow

load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="Agente conectado a LangChain vía backend.")

server = AgentServer()

@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    """
    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm=langchain.LLMAdapter(
            graph=create_workflow(),
        ),
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
    )
    """
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            voice="coral"
        )
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony() if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP else noise_cancellation.BVC(),
            ),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
