from threading import Thread
import litserve as ls
from config import MODEL
from transformers import (
    BitsAndBytesConfig,
    MllamaForCausalLM,
    TextIteratorStreamer,
    AutoTokenizer
)
import torch


class LlamaAPI(ls.LitAPI):
    def setup(self, device):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = MllamaForCausalLM.from_pretrained(
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

    
