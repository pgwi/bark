from typing import Optional
from scipy.io.wavfile import write as write_wav
from cog import BasePredictor, Input, Path, BaseModel
from bark import SAMPLE_RATE, generate_audio, preload_models, save_as_prompt
from bark.generation import ALLOWED_PROMPTS


class ModelOutput(BaseModel):
    prompt_npz: Optional[Path]
    audio_out: Path


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # for the pushed version on Replicate, the CACHE_DIR from bark/generation.py is changed to a local folder to
        # include the weights file in the image for faster inference
        preload_models()

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests "
            "such as playing tic tac toe.",
        ),
        history_prompt: str = Input(
            description="history choice for audio cloning, choose from the list",
            default=None,
            choices=sorted(list(ALLOWED_PROMPTS)),
        ),
        custom_history_prompt: Path = Input(
            description="Provide your own .npz file with history choice for audio cloning, this will override the "
            "previous history_prompt setting",
            default=None,
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
            description="return full generation as a .npz file to be used as a history prompt", default=False
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""

        if custom_history_prompt is not None:
            history_prompt = str(custom_history_prompt)

        audio_array = generate_audio(
            prompt,
            history_prompt=history_prompt,
            text_temp=text_temp,
            waveform_temp=waveform_temp,
            output_full=output_full,
        )
        output = "/tmp/audio.wav"
        if not output_full:
            write_wav(output, SAMPLE_RATE, audio_array)
            return ModelOutput(audio_out=Path(output))
        out_npz = "/tmp/prompt.npz"
        save_as_prompt(out_npz, audio_array[0])
        write_wav(output, SAMPLE_RATE, audio_array[-1])
        return ModelOutput(prompt_npz=Path(out_npz), audio_out=Path(output))
