from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from torch import nn
import einops


class Florence2Transform:

    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda().half().view(1, 1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).cuda().half().view(1, 1, 3, 1, 1)

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


class FlorenceEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        vlm_path = "microsoft/Florence-2-base"
        self.vlm = AutoModelForCausalLM.from_pretrained(
            vlm_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        prcsr = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
        self.tokenizer = prcsr.tokenizer
        del self.vlm.language_model.model.decoder
        del self.vlm.language_model.lm_head

    def _vlm_encode(self, image_features, text_embeds):
        """Run the main VLM transformer."""
        embeds, attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features, 
            text_embeds
        )

        encoder_outputs = self.vlm.get_encoder()(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=embeds,
            output_hidden_states=True,
            return_dict=True
        )

        return {
            'res2': encoder_outputs['hidden_states'][-4],
            'res3': encoder_outputs['hidden_states'][-3],
            'res4': encoder_outputs['hidden_states'][-2],
            'res5': encoder_outputs['hidden_states'][-1]
        }

    def _vit_encode(self, imgs):
        """
        Encode (multi-view) observations with ViT.

        Args:
            imgs: (B, n_views, C, H, W)
        Return:
            sequence: (B, n_feat, F)
                - F is the latent dimension (768)
                - n_feat = n_views * (1 + h'w') as cat((summ, unrolled))
                    where "1" indicates the position of the summary token
                    and h', w' the spatial dimensions of the feature map
                    that is unrolled to an 1d sequence.
                    (h', w') = (H, W) / 32
        """
        B, T, C, H, W = imgs.shape
        _features = self.vlm._encode_image(imgs.view(B * T, C, H, W))
        return einops.rearrange(_features, "(B T) N F -> B (T N) F", B=B, T=T)

    def _get_text_embeddings(self, text, device):
        """Get text embeddings to use with VLM"""
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)
        return self.vlm.get_input_embeddings()(text_inputs["input_ids"])

    def forward(self, img, text):
        """
        B, T, C, H, W = img.shape
        Returns B, N, F.
        """
        text_embeds = self._get_text_embeddings(text, img.device)
        image_features = self._vit_encode(img.half())
        vlm_features = self._vlm_encode(image_features, text_embeds)
        return vlm_features


def load_florence2():
    return FlorenceEncoder(), Florence2Transform()
