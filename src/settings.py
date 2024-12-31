from enum import Enum

import pycountry
from pydantic import BaseModel, Field, SecretStr, model_validator

from cat.mad_hatter.decorators import plugin, hook

from .transcribe import LocalWhisper

# Prepare language codes dictionary
language_codes = {
    lang.alpha_2.upper(): lang.alpha_2.lower()
    for lang in pycountry.languages
    if hasattr(lang, "alpha_2")
}

# Create LanguageCode enum from dictionary
LanguageCode = Enum("LanguageCode", language_codes, type=str)


class ModelSize(Enum):
    TINY = "tiny"
    TINY_EN = "tiny.en"
    BASE = "base"
    BASE_EN = "base.en"
    SMALL = "small"
    SMALL_EN = "small.en"
    MEDIUM = "medium"
    MEDIUM_EN = "medium.en"
    LARGE_V1 = "large-v1"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"
    LARGE = "large"
    OTHER = "Other"


class Device(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class ComputeType(Enum):
    INT8 = "int8"
    INT8_FLOAT32 = "int8_float32"
    INT8_FLOAT16 = "int8_float16"
    INT8_BFLOAT16 = "int8_bfloat16"
    INT16 = "int16"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


class Settings(BaseModel):
    language: LanguageCode = Field(
        title="Audio Language",
        description="Select the primary language of your audio files",
        default=LanguageCode.EN,
    )

    use_local_model: bool = Field(
        title="Use Offline Mode",
        description="Switch between offline (local) or online (OpenAI) transcription",
        default=True,
    )

    api_key: SecretStr = Field(
        title="OpenAI API Key",
        description="Required only for online mode - Get your key from platform.openai.com/api-keys",
        default="",
    )

    w_model_size: ModelSize = Field(
        title="Model Size",
        description="Larger models are more accurate but slower."\
        " The 'Other' option allows custom models, specify the path or Hugging Face model ID in 'Custom Model Path'",
        default=ModelSize.BASE,
    )

    device: Device = Field(
        title="Processing Device",
        description="Auto = best available, CPU = universal, CUDA = NVIDIA GPUs",
        default=Device.AUTO,
    )

    compute_type: ComputeType = Field(
        title="Performance Mode",
        description="FLOAT32 = most accurate, FLOAT16 = balanced, INT8 = fastest",
        default=ComputeType.FLOAT32,
    )

    w_model_path_or_id: str = Field(
        title="Custom Model Path",
        description="Optional: Custom model location or Hugging Face model ID, if 'Other' is selected in Model Size",
        default="",
    )

    audio_key: str = Field(
        title="Audio Input Source",
        description="WebSocket key for audio input (advanced setting)",
        default="audio",
    )

    @model_validator(mode="after")
    def validate_model_path_or_id(self):
        if self.w_model_size == ModelSize.OTHER and not self.w_model_path_or_id:
            raise ValueError("Custom Model Path is required when 'Other' is selected in Model Size")
        return self
    
    @model_validator(mode="after")
    def validate_api_key(self):
        if not self.use_local_model and not self.api_key:
            raise ValueError("OpenAI API Key is required for online mode")
        return self      


@plugin
def settings_model() -> Settings:
    return Settings


@plugin
def activated(plugin) -> None:
    """Setup the local model at plugin activation"""
    settings = plugin.load_settings()
    if settings and settings["use_local_model"]:
        LocalWhisper.get_instance(settings)


@hook(priority=0)
def after_cat_bootstrap(cat) -> None:
    """Setup the local model at startup"""
    settings = cat.mad_hatter.get_plugin().load_settings()
    if settings and settings["use_local_model"]:
        LocalWhisper.get_instance(settings)
