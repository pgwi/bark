from cog import BasePredictor, Input, Path
from scipy.io.wavfile import write as write_wav
from bark import SAMPLE_RATE, generate_audio, preload_models
from bark.generation import ALLOWED_PROMPTS


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # for the pushed version on Replicate, the CACHE_DIR from bark/generation.py is changed to a local folder to
        # contain the weights file in the image for faster inference
        preload_models()

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests "
            "such as playing tic tac toe.",
        ),
        history_prompt: str = Input(
            description="history choice for audio cloning",
            default=None,
            choices=list(ALLOWED_PROMPTS),
        ),
        text_temp: float = Input(
            description="generation temperature (1.0 more diverse, 0.0 more conservative)",
            default=0.7,
        ),
        waveform_temp: float = Input(
            description="generation temperature (1.0 more diverse, 0.0 more conservative)",
            default=0.7,
        ),
        output_full: bool = Input(
            description="return full generation to be used as a history prompt", default=False
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        audio_array = generate_audio(
            prompt,
            history_prompt=history_prompt,
            text_temp=text_temp,
            waveform_temp=waveform_temp,
            output_full=output_full,
        )
        output = "/tmp/audio.wav"
        write_wav(output, SAMPLE_RATE, audio_array)

        return Path(output)
