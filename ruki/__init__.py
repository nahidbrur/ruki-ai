"""Ruki: real-time Bangla speech-to-speech conversation.

Pipeline (one per WebSocket session), modelled on huggingface/speech-to-speech:

    mic (browser, 16 kHz) -> [VAD + denoise] -> [Hima-ASR] -> [Gemini, streamed] -> [OmniVoice TTS] -> speaker
"""

__version__ = "0.1.0"
