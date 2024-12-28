from cat.log import log
from cat.mad_hatter.decorators import hook

from .audio_parser import AudioParser
from .transcript import process_audio_file


@hook
def before_cat_reads_message(message_json, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()

    if settings == {}:
        log.error("No configuration found for WhisperingCat")
        return message_json

    if settings["audio_key"] not in message_json.keys():
        return message_json
    
    file_path = message_json[settings["audio_key"]]

    # Update the text in input
    message_json["text"] = process_audio_file(file_path, settings)
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

    settings = cat.mad_hatter.get_plugin().load_settings()

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
