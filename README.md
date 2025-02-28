[![awesome plugin](https://custom-icon-badges.demolab.com/static/v1?label=&message=awesome+plugin&color=383938&style=for-the-badge&logo=cheshire_cat_ai)](https://github.com/Furrmidable-Crew/WhisperingCat)

A plugin that adds voice recognition capabilities to Cheshire Cat by transcribing audio files from chat messages and file uploads.

## Features

- Transcribe audio from chat messages with `audio` field
- Process uploaded audio files automatically
- Support for both online (OpenAI API) and offline (Faster Whisper) transcription
- Multiple audio format support: `mp3`, `wav`, `m4a`, `mpga`, `ogg`, `webm`, `mpeg`, `mp4`

## Requirements

- Cheshire Cat >= 1.8.0
- For local transcription: sufficient disk space for model files
- For online transcription: OpenAI API key

## Installation

1. Go to your Cheshire Cat admin panel
2. Navigate to the "Plugin Store" section
3. Search for "Whispering Cat" and install
4. Configure the plugin settings after installation

## Configuration

In the settings panel, you can configure:

### Basic Settings
- **Audio Language**: Select the primary language of your audio files
- **Use Offline Mode**: Toggle between local transcription (Faster Whisper) or online (OpenAI)
- **OpenAI API Key**: Required only when using online mode

### Advanced Settings (Offline Mode)
- **Model Size**: Select model accuracy vs. speed (tiny, base, small, medium, large)
- **Processing Device**: Choose between CPU, CUDA (for NVIDIA GPUs), or Auto
- **Number of Workers**: Set parallel processing threads
- **Precision**: Balance between accuracy and speed
- **Custom Model Path**: Optional path for custom models

## Usage

### Chat Messages with Audio
Send audio in the `audio` field of your websocket message. Both URL links and Base64 encoded data are supported.

### File Uploads
Simply upload audio files in any supported format to have them automatically transcribed and stored in the decalrative memory.

## Troubleshooting

- **No transcription happening**: Check if the plugin is properly configured in settings
- **Error messages**: Verify your API key if using online mode
- **Performance issues**: Try a smaller model size or reduce workers if using local mode