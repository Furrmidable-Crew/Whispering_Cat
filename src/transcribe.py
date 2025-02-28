import os
import re
import base64
import requests
from io import BytesIO
from urllib.parse import urlparse
from typing import Dict, Tuple, Union, BinaryIO

from .local_whisper import LocalWhisper


def process_audio_file(file_path: str, settings: Dict) -> str:
    # Get the file handle and MIME type
    file_info: Tuple[BinaryIO, str] = _get_file_handle(file_path)

    file, mime_type = file_info

    if settings.get("use_local_model"):
        transcription = _transcribe_local(
            file=file,
            settings=settings 
        )
    else: 
        if not settings.get("api_key"):
            raise ValueError("API key is required for OpenAI's transcription API")

        transcription = _transcribe(
            file=(file.name, file.read(), mime_type),
            settings=settings
        )

    # Closing the file even if BytesIO doesn't 
    # require it to free up resources
    file.close()

    return transcription


def _transcribe(file: Tuple[str, bytes, str], settings: Dict) -> str:
    key = settings["api_key"]
    lang = settings["language"]

    if not key:
        raise ValueError("API key is required for OpenAI's transcription API")

    # TODO: Split the file if it exceeds the maximum limit and transcribe in parts

    if len(file[1]) > 25 * 1000000:
       raise ValueError("File size exceeds the maximum limit of 25MB")
    res = requests.post("https://api.openai.com/v1/audio/transcriptions", 
        headers = {
            "Authorization": f"Bearer {key}"
        }, files = {
            "file": file
        }, data = {
            "model": "whisper-1",
            "language": lang
        }
    )

    json = res.json()

    if res.status_code != 200:
        error = json.get("error").get("message")
        raise ValueError(f"Failed to transcribe the audio: {error}")

    return json['text']


def _transcribe_local(file: BinaryIO, settings: Dict) -> str:
    whisper = LocalWhisper.get_instance(settings)
    segment, _ = whisper.transcribe(
        file, 
        language=settings["language"], 
        multilingual=True,
        vad_filter=True # Remove silence
    )
    result = "".join([str(s.text) for s in segment])
    return result


def _get_file_handle(file_path: str) -> Union[Tuple[BinaryIO, str], None]:
    # Check if it's a data URI
    if str(file_path).startswith("data:"):
        return _handle_data_uri(file_path)
    
    parsed_file = urlparse(file_path) 
    is_url = all([parsed_file.scheme, parsed_file.netloc])
    
    # Check if it's an URL or a local file
    if is_url:
        return _handle_url(file_path)
    else:
        return _handle_local_file(file_path)


def _handle_data_uri(data_uri: str) -> Union[Tuple[BinaryIO, str], None]:
    mime_match = re.match(r'data:audio/([a-zA-Z0-9]+);base64,', data_uri)
    if not mime_match:
        raise ValueError("Invalid data URI")

    # Map of valid MIME types to file extensions for Whisper
    mime_to_ext = {
        'wav': 'wav',
        'mpeg': 'mp3',
        'mp4': 'm4a',
        'ogg': 'ogg',
        'webm': 'webm'
    }

    mime_type = mime_match.group(1)
    file_type = mime_to_ext.get(mime_type)
    if not file_type:
        raise ValueError(f"Unsupported MIME type: {mime_type}")

    # Extract base64 data and decode it
    base64_data = data_uri.split(',')[1]
    audio_bytes = base64.b64decode(base64_data)
    
    # Using BytesIO to create a file-like object
    file = BytesIO(audio_bytes)
    file.name = f'audio.{file_type}'
    
    return file, mime_type


def _handle_url(url: str) -> Union[Tuple[BinaryIO, str], None]:
    res = requests.get(url)

    if res.status_code != 200:
        raise ValueError(f"Failed to fetch the file from {url}")

    file = BytesIO(res.content)
    file.name = os.path.basename(url)
    mime_type = res.headers.get('Content-Type')

    return file, mime_type


def _handle_local_file(file_path: str) -> Union[Tuple[BinaryIO, str], None]:
    with open(file_path, 'rb') as f:
        content = f.read()
    
    file = BytesIO(content)
    file.name = os.path.basename(file_path)

    ext_to_mime = {
        'wav': 'audio/x-wav',
        'mp3': 'audio/mpeg',
        'm4a': 'audio/mp4',
        'ogg': 'audio/ogg',
        'webm': 'audio/webm'
    }

    file_ext = file.name.split('.')[-1]

    if file_ext not in ext_to_mime.keys():
        raise ValueError(f"Unsupported file type: {file_ext}")
    
    mime_type = "audio/" + ext_to_mime.get(file_ext)

    return file, mime_type
