import os
import re
import base64
import requests
from io import BytesIO
from tempfile import TemporaryFile
from urllib.parse import urlparse
from typing import Dict, Tuple, Union, BinaryIO


def process_audio_file(file_path: str, settings: Dict) -> str:
    # Get the file handle
    file_info = _get_file_handle(file_path)
    if not file_info:
        raise ValueError("Unsupported file type or invalid file path.")
    
    file, mime_type = file_info

    return transcript(
        key=settings["api_key"],
        lang=settings["language"],
        file=(file.name, file.read(), mime_type)
    )


def transcript(key, lang, file):
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
    
    return json['text']


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
    
    # Create a file-like object from the decoded audio data
    file = BytesIO(audio_bytes)
    file.name = f'audio.{file_type}'
    
    return file, mime_type


def _handle_url(url: str) -> Union[Tuple[BinaryIO, str], None]:
    res = requests.get(url)

    if res.status_code != 200:
        raise ValueError(f"Failed to fetch the file from {url}")

    file = TemporaryFile('wb+')
    # Write the file content to a temporary file
    file.write(res.content)
    # Reset the file pointer to the beginning
    file.seek(0)
    file.name = os.path.basename(url)
    # Get the MIME type and file extension
    mime_type = res.headers.get('Content-Type')

    return file, mime_type


def _handle_local_file(file_path: str) -> Union[Tuple[BinaryIO, str], None]:
    file = open(file_path)
    file.name = os.path.basename(file_path)
    file_ext = os.path.splitext(file.name)[1][1:]
    mime_type = f'audio/{file_ext}'

    return file, mime_type