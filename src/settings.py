from enum import Enum
import pycountry
from pydantic import BaseModel, Field, SecretStr
from cat.mad_hatter.decorators import plugin

# Prepare language codes dictionary
language_codes = {
    lang.alpha_2.upper(): lang.alpha_2.lower()
    for lang in pycountry.languages
    if hasattr(lang, "alpha_2")
}

# Create LanguageCode enum from dictionary
LanguageCode = Enum("LanguageCode", language_codes, type=str)

class Settings(BaseModel):
    api_key: SecretStr = Field(
        title="API Key", 
        description="The API key for OpenAI's transcription API.", 
        default=""
    )
    language: LanguageCode = Field(
        title="Language", 
        description="The language of the audio file in ISO-639-1 format. Defaults to 'en' (English).", 
        default=LanguageCode.EN
    )
    audio_key: str = Field(
        title="Audio Key", 
        description="The key for the WebSocket object to recognize additional content. Defaults to 'audio'.", 
        default="audio"
    )

@plugin
def settings_model():   
    return Settings