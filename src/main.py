from cat.log import log
from cat.mad_hatter.decorators import hook
from cat.convo.messages import UserMessage

from .audio_parser import AudioParser
from .transcribe import process_audio_file


@hook
def before_cat_reads_message(message: UserMessage, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()

    if settings == {}:
        log.error("No configuration found for WhisperingCat")
        return

    if message.audio is None:
        log.debug("No audio found in the message")
        return
        
    try:
        # Update the text in input whis the transcribed audio
        message.text = process_audio_file(message.audio, settings)
    except ValueError as e:
        log.error(str(e))
        cat.send_ws_message(str(e), "notification")
    except Exception as e:
        log.error(f"An error occurred while processing the audio: {e}")
        cat.send_ws_message("An error occurred while processing the audio.", "error")

    return message

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

    # Check if the plugin have settings
    # And that the API key is set if the local model is not used
    if not settings:
        raise ValueError("No configuration found for WhisperingCat")
    if not settings["api_key"] and not settings["use_local_model"]:
        raise ValueError("API key is required for OpenAI's transcription API, please set it in the plugin settings or use a local model")

    new_file_handlers["video/mp4"] = AudioParser(settings)
    new_file_handlers["audio/ogg"] = AudioParser(settings)
    new_file_handlers["audio/wav"] = AudioParser(settings)
    new_file_handlers["audio/webm"] = AudioParser(settings)
    new_file_handlers["audio/mpeg"] = AudioParser(settings)
    new_file_handlers["audio/x-wav"] = AudioParser(settings)

    return new_file_handlers
