import tempfile
import requests
from typing import Iterator
from langchain.schema import Document
from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.base import BaseBlobParser
from faster_whisper import WhisperModel


def transcript(key, lang, file):
    """Transcribe an audio file using OpenAI."""
    if len(file[1]) > 25 * 1000000:
        return "The audio is too large for OpenAI to process."
    res = requests.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {key}"},
        files={"file": file},
        data={"model": "whisper-1", "language": lang},
    )
    json = res.json()
    return json["text"]


def transcript_local(audio_file, model_size_or_path, device, compute_type, language):
    """Transcribe an audio file using local model."""
    model = WhisperModel(model_size_or_path, device=device, compute_type=compute_type)
    result = model.transcribe(audio_file, language=language)
    return result


class AudioParser(BaseBlobParser):
    """Parser for audio blobs."""

    def __init__(self, settings):
        """Initialize the parser."""
        self.key = settings.get("api_key")
        self.lang = settings.get("language", "en")
        self.use_local_model = settings.get("use_local_model", True)
        self.model_size_or_path = settings.get("model_size_or_path", "large-v3")
        self.device = settings.get("device", "cpu")
        self.compute_type = settings.get("compute_type", "int8")

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        if not self.use_local_model:
            content = transcript(
                self.key, self.lang, (blob.path, blob.as_bytes(), blob.mimetype)
            )
        else:

            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(blob.as_bytes())
                temp_file_path = temp_file.name
                segments, _ = transcript_local(
                    temp_file_path,
                    model_size_or_path=self.model_size_or_path,
                    device=self.device,
                    compute_type=self.compute_type,
                    language=self.lang,
                )
                content = "".join([s.text for s in segments])

        yield Document(
            page_content=content,
            metadata={"source": "whispering_cat", "name": blob.path.rsplit(".", 1)[0]},
        )
