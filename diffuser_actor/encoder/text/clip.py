"""
Precompute embeddings of instructions.
"""
import transformers
import torch
import torch.nn as nn


class ClipTextEncoder(nn.Module):

    def __init__(self, text_max_length=53):
        super().__init__()
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.tokenizer.model_max_length = text_max_length
        self.model = transformers.CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    @torch.inference_mode()
    def forward(self, instructions, device):
        tokens = self.tokenizer(instructions, padding="max_length")["input_ids"]
        tokens = torch.tensor(tokens).to(device)
        encodings = self.model(tokens).last_hidden_state

        return encodings
