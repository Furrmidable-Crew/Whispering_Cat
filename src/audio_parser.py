from typing import Iterator
from langchain.schema import Document
from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.base import BaseBlobParser

from .transcript import transcript

class AudioParser(BaseBlobParser):
    """Parser for audio blobs."""

    def __init__(self, key: str, lang: str = "en"):
        self.key = key
        self.lang = lang

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
    
        content = transcript(self.key, self.lang, (blob.path, blob.as_bytes(), blob.mimetype))

        yield Document(page_content=content, metadata={"source": "whispering_cat", "name": blob.path.rsplit('.', 1)[0]})
