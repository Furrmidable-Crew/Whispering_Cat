from typing import Iterator
from tempfile import NamedTemporaryFile

from langchain.schema import Document
from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.base import BaseBlobParser

from cat.log import log

from .transcribe import process_audio_file


class AudioParser(BaseBlobParser):
    """Parser for audio blobs."""

    def __init__(self, settings: dict):
        self.settings = settings

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""

        file_ext = blob.path.rsplit(".", 1)[1]
        
        with NamedTemporaryFile(suffix=f".{file_ext}") as tmp_file:
            tmp_file.write(blob.data)
            tmp_file.seek(0)

            # Transcribe the audio
            log.debug(f"Transcribing audio file: {blob.path}")
            content = process_audio_file(tmp_file.name, self.settings)
      
        yield Document(
            page_content=content,
            metadata={"source": "whispering_cat", "name": blob.path.rsplit(".", 1)[0]},
        )
