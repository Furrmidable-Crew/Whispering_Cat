from typing import Iterator
from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document
import requests

class AudioParser(BaseBlobParser):
    """Parser for audio blobs."""

    def __init__(self, key: str, lang: str = "en"):
        self.key = key
        self.lang = lang

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""

        content = ""
        # TODO: Check if the file is too large for OpenAI (max 25mb)
        with blob.as_bytes_io() as file:
            file_name, file_extension = blob.path.rsplit('.', 1)
            res = requests.post("https://api.openai.com/v1/audio/transcriptions", 
                                headers={
                                    "Authorization": f"Bearer {self.key}"
                                }, files={
                                    "file": (f"audio.{file_extension}", file, blob.mimetype),
                                    "model": (None, "whisper-1"),
                                    "language": (None, self.lang)
                                })
            json = res.json()
            content = json['text']
    

        yield Document(page_content=content, metadata={"source": "whispering_cat", "name": blob.path.rsplit('.', 1)[0]})
