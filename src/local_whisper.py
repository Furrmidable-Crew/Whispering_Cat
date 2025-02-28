import threading
from typing import Dict

from faster_whisper import WhisperModel
from faster_whisper.utils import download_model
from huggingface_hub.file_download import LocalEntryNotFoundError

from cat.log import log

class LocalWhisper:
    """Singleton class to manage the local Whisper model instance."""
   
    __whisper = None
    download_path = None
    __current_settings = None
    __lock = threading.Lock()
    
    @staticmethod
    def get_instance(settings: Dict):
        """Get the Whisper model instance."""
        # Lock to prevent multiple threads from creating the model
        # at the same time
        with LocalWhisper.__lock:
            # Return the existing instance if the model is already loaded 
            if (LocalWhisper.__whisper is not None) and (LocalWhisper.__current_settings == settings):
                return LocalWhisper.__whisper
        
            LocalWhisper.__current_settings = settings
            LocalWhisper.create_new_whisper(settings)
            return LocalWhisper.__whisper
       
    @staticmethod
    def create_new_whisper(settings: Dict):
        """Create a new Whisper model instance."""
        model = LocalWhisper._get_model_id(settings)
                
        # Download the model if it's not already downloaded
        if not LocalWhisper.is_model_downloaded(model):
            log.info(f"Downloading Whisper model `{model}`...")
            download_model(
                model,
                cache_dir=LocalWhisper.download_path
            )

        # Free up resources by unloading the previous model
        old_whisper = LocalWhisper.__whisper
        del old_whisper

        log.info(f"Loading local Whisper model `{model}`...")
        # Load the new model
        LocalWhisper.__whisper = WhisperModel(
            model_size_or_path=model,
            device=settings["device"],
            compute_type=settings["compute_type"],
            num_workers=settings["n_workers"],
            download_root=LocalWhisper.download_path,
            local_files_only=True
        )
    
    @staticmethod
    def is_model_downloaded(model: str) -> bool:
        try:
            # Check if the model is already downloaded
            download_model(
                model,
                local_files_only=True,
                cache_dir=LocalWhisper.download_path,
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

