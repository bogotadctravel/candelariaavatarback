from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv


# Carga las variables del archivo .env al inicio de la aplicación
load_dotenv()
# Accede a las variables de entorno
SECRET_KEY_VALUE = os.getenv("SECRET_KEY")
ACCESS_TOKEN_EXPIRE_MINUTES_VALUE = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")

class Settings(BaseSettings):
    SECRET_KEY: str = SECRET_KEY_VALUE
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = ACCESS_TOKEN_EXPIRE_MINUTES_VALUE

settings = Settings()
