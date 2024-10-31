from threading import Thread
import litserve as ls
from litserve.specs.openai import ChatCompletionRequest, ChatMessage
from config import MODEL
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    AutoTokenizer
)
import torch
from TTS.api import TTS
import numpy as np
import scipy
import io


class LlamaAPI(ls.LitAPI):
    def setup(self, device):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
            device_map=device,
            quantization_config=quantization_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        self.device = device

    def decode_request(self, request: ChatCompletionRequest, context: dict):
        context["model_args"] = {
            "temperature": request.temperature or 0.7,
            "top_p": request.top_p or 0.9,
            "max_new_tokens": request.max_tokens or 2048
        }
        input = self.tokenizer.apply_chat_template(request.messages, add_generation_prompt=True, return_tensors="pt", tokenize=True)
        return input

    def predict(self, inputs, context: dict):
        model_kwargs = {
            "input_ids": inputs.to(self.device),
            "streamer": self.streamer,
            "eos_token_id": self.tokenizer.eos_token_id,
            **context["model_args"]
        }
        thread = Thread(target=self.model.generate, kwargs=model_kwargs)
        thread.start()
        for i in self.streamer:
            yield i


    def encode_response(self, outputs, context: dict):
        for output in outputs:
            if self.tokenizer.eos_token in output:
                output = output.replace(self.tokenizer.eos_token, "")
            yield ChatMessage(role="assistant", content=output)


class TTSAPI(ls.LitAPI):
    def setup(self, device):
        self.device = device
        self.tts = TTS("tts_models/en/vctk/vits").to(device)

    def decode_request(self, request, context: dict):
        # context["text"] = request.get("text")
        # context["speaker_wav"] = "aizen.wav"
        # context["language"] = "ja"
        context["text"] = request.get("text")
        context["speaker"] = "243"
        yield context

    def predict(self, inptus, context: dict):
        text = context["text"]
        speaker = context["speaker"]

        wav = self.tts.tts(text=text, speaker=speaker)
        wav = np.int16(wav / np.max(np.abs(wav)) * 32767)
        sample_rate = 24000
        byte_data = io.BytesIO()
        scipy.io.wavfile.write(byte_data, sample_rate, wav)
        byte_data.seek(0)
        
        yield byte_data

    def encode_request(self, outputs, context: dict):
        return outputs

