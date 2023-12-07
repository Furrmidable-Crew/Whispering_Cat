import os
import requests
from urllib.parse import urlparse
from pydantic import BaseModel, Field
from tempfile import TemporaryFile
from cat.log import log
from cat.mad_hatter.decorators import hook, plugin
from .audio_parser import AudioParser, transcript


class Settings(BaseModel):
    api_key: str = Field(title="API Key", description="The API key for OpenAI's transcription API.", default="")
    language: str = Field(title="Language", description="The language of the audio file in ISO-639-1 format. Defaults to 'en' (English).", default="en")
    audio_key: str = Field(title="Audio Key", description="The key for the WebSocket object to recognize additional content. Defaults to 'whispering_cat'.", default="whispering_cat")

@plugin
def settings_schema():   
    return Settings.schema()

@hook
def before_cat_reads_message(message_json, cat):
    settings = cat.mad_hatter.plugins["whispering_cat"].load_settings()

    if settings == {}:
        log.error("No configuration found for WhisperingCat")
        return message_json

    if settings["audio_key"] not in message_json.keys():
        return message_json
    
    file_path = message_json[settings["audio_key"]]

    # Check if it's an url
    parsed_file = urlparse(file_path)
    is_url = all([parsed_file.scheme, parsed_file.netloc])

    if is_url:
        # Get the file
        res = requests.get(file_path)
        # write it in a temporary file
        file = TemporaryFile('wb+')
        file.write(res.content)
        file.seek(0)
    else:
        # Othewhise open the file 
        file = open(file_path)
        
    # Get the file type
    file_type = os.path.splitext(os.path.basename(file_path))[1]

    # Making the transcription 
    transcription = transcript(
        key=settings["api_key"],
        lang=settings["language"],
        file=(file_type, file.read())
    )   

    # Update the text in input
    message_json["text"] = transcription
    return message_json


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
        log.error("No configuration found for WhisperingCat")
        cat.send_ws_message("You did not configure the API key for the transcription API!", "notification")
        return

    new_file_handlers["video/mp4"] = AudioParser(settings["api_key"], settings["language"])
    new_file_handlers["audio/ogg"] = AudioParser(settings["api_key"], settings["language"])
    new_file_handlers["audio/wav"] = AudioParser(settings["api_key"], settings["language"])
    new_file_handlers["audio/webm"] = AudioParser(settings["api_key"], settings["language"])
    new_file_handlers["audio/mpeg"] = AudioParser(settings["api_key"], settings["language"])
    new_file_handlers["audio/x-wav"] = AudioParser(settings["api_key"], settings["language"])

    return new_file_handlers
