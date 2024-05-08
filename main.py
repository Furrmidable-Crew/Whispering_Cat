import os
import requests
from urllib.parse import urlparse
from pydantic import BaseModel, Field
from tempfile import TemporaryFile
from cat.log import log
from cat.mad_hatter.decorators import hook, plugin
from .audio_parser import AudioParser, transcript, transcript_local


class Settings(BaseModel):
    use_local_model: bool = Field(
        title="Use local model",
        description="Whether to use a local model (true) or OpenAI API (false)?.",
        default=True,
    )
    api_key: str = Field(
        title="API Key",
        description="The API key for OpenAI's transcription API.",
        default="",
    )
    language: str = Field(
        title="Language",
        description="The language of the audio file in ISO-639-1 format. Defaults to 'en' (English).",
        default="en",
    )
    audio_key: str = Field(
        title="Audio Key",
        description="The key for the WebSocket object to recognize additional content. Defaults to 'whispering_cat'.",
        default="whispering_cat",
    )
    model_size_or_path: str = Field(
        title="Model size or path",
        description="Size of the model to use (tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3, or large), a path to a converted model directory, or a CTranslate2-converted Whisper model ID from the HF Hub. When a size or a model ID is configured, the converted model is downloaded from the Hugging Face Hub.",
        default="large-v3",
    )
    device: str = Field(
        title="Device",
        description='Device to use for computation ("cpu", "cuda", "auto").',
        default="cpu",
    )
    compute_type: str = Field(
        title="Compute type",
        description="Type to use for computation. See https://opennmt.net/CTranslate2/quantization.html.",
        default="int8",
    )


@plugin
def settings_schema():
    return Settings.schema()


@hook
def before_cat_reads_message(message_json, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()

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
        file = TemporaryFile("wb+")
        file.write(res.content)
        file.seek(0)
    else:
        # Othewhise open the file
        file = open(file_path)

    # Get the file type
    file_type = os.path.splitext(os.path.basename(file_path))[1]

    if not settings.get("use_local_model", False):
        # Making the transcription
        transcription = transcript(
            key=settings.get("api_key"),
            lang=settings.get("language", "en"),
            file=(file_type, file.read()),
        )
    else:
        # Making the transcription using the local model
        transcription = transcript_local(
            file_path,
            model_size_or_path=settings.get("model_size_or_path", "large-v3"),
            device=settings.get("device", "cpu"),
            compute_type=settings.get("compute_type", "int8"),
            language=settings.get("language", "en"),
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
        cat.send_ws_message(
            f"""The audio \"`{name}`\" says:
                            \"{content}\"""",
            "chat",
        )

    return text


@hook
def rabbithole_instantiates_parsers(file_handlers: dict, cat) -> dict:
    new_file_handlers = file_handlers

    settings = cat.mad_hatter.get_plugin().load_settings()

    if settings == {}:
        log.error("No configuration found for WhisperingCat")
        cat.send_ws_message(
            "You did not configure the API key for the transcription API!",
            "notification",
        )
        return

    new_file_handlers["video/mp4"] = AudioParser(settings)
    new_file_handlers["audio/ogg"] = AudioParser(settings)
    new_file_handlers["audio/wav"] = AudioParser(settings)
    new_file_handlers["audio/webm"] = AudioParser(settings)
    new_file_handlers["audio/mpeg"] = AudioParser(settings)
    new_file_handlers["audio/x-wav"] = AudioParser(settings)

    return new_file_handlers
