from typing import Dict
from pathlib import Path

from faster_whisper import WhisperModel
from faster_whisper.utils import download_model
from huggingface_hub.file_download import LocalEntryNotFoundError

from cat.log import log
from cat.utils import get_base_path


class LocalWhisper:
    """Singleton class to manage the local Whisper model instance."""
   
    whisper = None
    current_model = None
    download_path = Path(get_base_path()) / "data" / "whispering_cat" / "models"

    @staticmethod
    def get_instance(settings: Dict):
        """Get the Whisper model instance."""
        model = LocalWhisper._get_model_id(settings)
        device = settings["device"]
        compute_type = settings["compute_type"]

        # Return the existing instance if the model is already loaded 
        if (LocalWhisper.whisper is not None) and (LocalWhisper.current_model == model):
            return LocalWhisper.whisper
        
        # Download the model if it's not already downloaded
        if not LocalWhisper._model_downloaded(model):
            log.info(f"Downloading Whisper model `{model}`...")
            download_model(
                model,
                cache_dir=LocalWhisper.download_path
            )

        # Update the current model
        LocalWhisper.current_model = model

        # Free up resources by unloading the previous model
        old_whisper = LocalWhisper.whisper
        del old_whisper

        # Load the new model
        log.info(f"Loading local Whisper model `{model}`...")
        LocalWhisper.whisper = WhisperModel(
            model_size_or_path=model,
            device=device,
            compute_type=compute_type,
            download_root=LocalWhisper.download_path
        )

        return LocalWhisper.whisper
    
    @staticmethod
    def _model_downloaded(model: str) -> bool:
        try:
            # Check if the model is already downloaded
            download_model(
                model,
                local_files_only=True,
                cache_dir=LocalWhisper.download_path
            )
            return True
        except LocalEntryNotFoundError:
            return False

    @staticmethod
    def _get_model_id(settings: Dict) -> str:
        # Get the correct model size, path or Hugging Face model ID    
        model = settings["w_model_size"]
        if model == "other":
            model = settings["w_model_path_or_id"]
        return model

