import transformers
import torch
import torch.nn as nn


class ClipTextEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.model = transformers.CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    @torch.inference_mode()
    def forward(self, instructions, device):
        tokens = self.tokenizer(instructions, padding="longest")["input_ids"]
        tokens = torch.tensor(tokens).to(device)
        encodings = self.model(tokens).last_hidden_state

        return encodings
