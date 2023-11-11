from cat.mad_hatter.decorators import hook, plugin
from cat.plugins.whispering_cat.parser import AudioParser
from pydantic import BaseModel, Field

class Settings(BaseModel):
    api_key: str = Field(title="API Key", description="The API key for OpenAI's transcription API.", default="")
    language: str = Field(title="Language", description="The language of the audio file in ISO-639-1 format. Defaults to English (en).", default="en")

@plugin
def settings_schema():   
    return Settings.schema()

@hook
def before_rabbithole_splits_text(text: list, cat):
    
    is_audio = text[0].metadata["source"] == "whispering_cat"

    if is_audio:
        content = text[0].page_content
        name = text[0].metadata["name"]
        cat.send_ws_message(f"""The audio \"`{name}`\" says:
                            \"{content}\"""", "chat")

    return text

@hook
def rabbithole_instantiates_parsers(file_handlers: dict, cat) -> dict:
    new_file_handlers = file_handlers

    settings = cat.mad_hatter.plugins["whispering_cat"].load_settings()

    if settings == {}:
        settings = {
            "api_key": "",
            "language": "en"
        }

    new_file_handlers["audio/mpeg"] = AudioParser(settings["api_key"], settings["language"])
    new_file_handlers["audio/webm"] = AudioParser(settings["api_key"], settings["language"])
    new_file_handlers["audio/wav"] = AudioParser(settings["api_key"], settings["language"])
    new_file_handlers["audio/ogg"] = AudioParser(settings["api_key"], settings["language"])
    new_file_handlers["video/mp4"] = AudioParser(settings["api_key"], settings["language"])

    return new_file_handlers