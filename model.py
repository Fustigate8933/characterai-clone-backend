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
        tts = TTS("tts_models/en/vctk/vits").to(device)

    def decode_request(self, request, context: dict):
        # context["text"] = request.get("text")
        # context["speaker_wav"] = "aizen.wav"
        # context["language"] = "ja"
        context["text"] = request.get("text")
        context["speaker"] = "243"
        return context

    def predict(self, inptus, context: dict):
        text = context["text"]
        speaker = context["speaker"]

        wav = self.tts.tts(text=text, speaker=speaker)
        wav = np.int16(wav / np.max(np.abs(wav)) * 32767)
        sample_rate = 24000
        byte_data = io.BytesIO()
        scipy.io.wavfile.write(byte_data, sample_rate, wav)
        byte_data.seek(0)
        
        return byte_data

    def encode_request(self, outputs, context: dict):
        return outputs


if __name__ == "__main__":
    import argparse
    from openai import OpenAI

    client = OpenAI(
        base_url="http://0.0.0.0:8000/v1",
        api_key="stoopid"
    )

    def get_response(message):
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a specialist on Japan related matter, including tourism spots, history, langauge, food, and more."
                },
                {
                    "role": "user",
                    "content": "Can you give me some examples of Japanese omatsuri?"
                },
                {
                    "role": "assistant",
                    "content": """
                        1. **Gion Matsuri** (July 17th, Kyoto): A colorful festival to honor the gods and ward off evil spirits. It features traditional music, dances, and parades.
                        2. **Sanja Matsuri** (May 15th, Tokyo): A lively festival at the Asakusa Shrine, where priests carry portable shrines and perform traditional dances.
                        3. **Star Festival (Tanabata Matsuri)** (July 7th, Sendai): A beautiful festival inspired by the Chinese legend of two star-crossed lovers. It features lanterns, poems, and traditional sweets.
                        4. **Miyajima Matsuri** (August 25th, Hiroshima): A festival on Miyajima Island, where people carry shrines and perform rituals to honor the god of the sea.
                        These are just a few examples of the many omatsuri celebrations held throughout Japan. Each festival has its unique characteristics, traditions, and cultural significance, making them an integral part of Japanese culture and society.%
                    """
                },
                {
                    "role": "user",
                    "content": "Can you elaborate in one paragraph about the 3rd one you have mentioned?"
                },
                {
                    "role": "assistant",
                    "content": """
                    The Star Festival, also known as Tanabata Matsuri, is a magical and romantic celebration inspired by the ancient Chinese legend of two star-crossed lovers, Orihime and Hikoboshi. According to the myth, the lovers were separated by the Milky Way and could only meet once a year on the seventh day of the seventh month, which falls on July 7th in the lunar calendar. In Japan, this festival is celebrated on July 7th or around that date, with festivities typically held in the evening. People write wishes and poems on tanzaku papers and tie them to bamboo trees, while thousands of colorful paper lanterns are hung to represent the stars in the sky. The atmosphere is filled with music, dancing, and the scent of traditional sweets, creating a whimsical and enchanting experience that is a must-see for anyone visiting Japan during this time.%
                    """
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            max_tokens=512,
            stream=True
        )
        for chunk in stream:
            print(chunk.choices[0].delta.content or '', flush=True, end="")

    parser = argparse.ArgumentParser("Chat with Llama")
    parser.add_argument("text", help="Your message to Llama")
    args = parser.parse_args()

    get_response(args.text)
