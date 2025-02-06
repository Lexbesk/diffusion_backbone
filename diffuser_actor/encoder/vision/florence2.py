from transformers import AutoModelForCausalLM, AutoProcessor
import torch
import torch.nn as nn
import torch.nn.functional as F


class AggregatorTransformer(nn.Module):
    def __init__(self, d_model=768, n_heads=12, num_layers=2):
        super().__init__()
        # A single aggregator token. Shape (1, 1, d_model).
        self.agg_token = nn.Parameter(torch.randn(1, 1, d_model))

        # A simple embedding projection (dummy in this example).
        # In practice, you'd have an embedding for your tokens or patches.
        self.input_projection = nn.Linear(d_model, d_model)

        # A transformer encoder with `num_layers` blocks.
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        """
        x: (B, T, d_model)
        """
        B, T, D = x.shape
        
        # 1. Project the inputs (optional if x is already d_model dimension)
        x = self.input_projection(x)  # still (B, T, D)

        # 2. Expand aggregator token to match batch size
        # shape: (B, 1, D)
        agg_token_expanded = self.agg_token.expand(B, -1, D)

        # 3. Concat aggregator token to the *start* (or end) of the sequence
        # shape: (B, T+1, D)
        x_with_agg = torch.cat((agg_token_expanded, x), dim=1)

        # The PyTorch transformer wants shape (T, B, D) -- (seq_len, batch, dim)
        x_trans = x_with_agg.transpose(0, 1)  # (T+1, B, D)

        # 4. Pass through the Transformer
        # By default, no special mask means everything can attend to everything
        encoded = self.transformer(x_trans)  # (T+1, B, D)

        # 5. Extract the aggregator token's final state (the "summary")
        agg_output = encoded[0]  # shape: (B, D), the aggregator is at index 0

        # 6. Project to final dimension
        # agg_output = self.final_projection(agg_output)  # shape: (B, final_dim)

        # (Optionally) do something with that aggregator output, like a classifier head
        return agg_output

class GripperEncoder(nn.Module):
    # takes in a gripper of shape (B, nhist, D), passes though MLP and returns something of shape (B, F)
    def __init__(self, D, H, F):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(D, H),
            nn.ReLU(),
            nn.Linear(H, F)
        )

    def forward(self, gripper):
        return self.mlp(gripper)
        


class FlorenceEncoder:

    def __init__(self, device, aggregate=True, d_agg=768, agg_heads=12, num_agg_layers=2):
        vlm_path = "microsoft/Florence-2-base"
        self.device = device
        self.vlm = AutoModelForCausalLM.from_pretrained(vlm_path, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

        del self.vlm.language_model.model.decoder
        del self.vlm.language_model.lm_head
        
        self.aggregate = aggregate
        if self.aggregate:
            # add aggregator transformer
            self.aggregator = AggregatorTransformer(d_model=d_agg, n_heads=agg_heads, num_layers=num_agg_layers).to(self.device)
        else:
            self.aggregator = None

        self.gripper_encoder = GripperEncoder(10, 256, 64).to(self.device)



        # make all vlm parameters non-trainable
        for param in self.vlm.parameters():
            param.requires_grad = False
        
        # make all aggregator parameters trainable
        if self.aggregate:
            for param in self.aggregator.parameters():
                param.requires_grad = True

        # make all gripper encoder parameters trainable
        for param in self.gripper_encoder.parameters():
            param.requires_grad = True
        

    def _pad_or_truncate_sequence(self, sequence, target_len):
        if sequence.shape[1] > target_len:
            truncation_ratio = (sequence.shape[1] - target_len) / sequence.shape[1]
            if truncation_ratio > 0.5:  # More than 50% truncation
                logger.warning(f"Severe sequence truncation: {truncation_ratio:.2%}")
            return sequence[:, :target_len, :]
        return F.pad(sequence, (0, 0, 0, target_len - sequence.shape[1], 0, 0))

    def encode(self, image_tensor, text, curr_gripper):
        text_embeds = self._get_text_embeddings(text, self.device)
        B, T, C, H, W = image_tensor.shape
        image_features = self._vit_encode(image_tensor, B, T, C, H, W, self.device)
        vlm_features = self._vlm_encode(image_features, text_embeds)
        gripper_features = self.gripper_encoder(curr_gripper)

        # shape of gripper_features: (B, T, F)
        # change to (B, F) by averaging over time
        gripper_features = torch.mean(gripper_features, dim=1)

        # concatenate all features
        return torch.cat([gripper_features, vlm_features], dim=1)


    def _vlm_encode(self, image_features, text_embeds):
            """Process features through the VLM's transformer"""

            # 50 tokens per image, and these tokens come first
            merged_embeds, attention_mask = self.vlm._merge_input_ids_with_image_features(
                image_features, 
                text_embeds
            )
            
            encoder_outputs = self.vlm.get_encoder()(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=merged_embeds,
                return_dict=True
            )
            
            vlm_encoding = self._pad_or_truncate_sequence(
                encoder_outputs.last_hidden_state,
                target_len=75
            )

            if not self.aggregate:
                return vlm_encoding
            return self.aggregator(vlm_encoding)

    def _vit_encode(self, view_tensor, B, T, C, H, W, device):
            """Encode a view through just the ViT"""
            view_features = self.vlm._encode_image(
                view_tensor.to(device).view(B * T, C, H, W)
            )
            # view_features = view_features + self.static_view_embedding.expand_as(view_features)
            return view_features.view(B, T*view_features.shape[1], -1)

    def _get_text_embeddings(self, text, device):
            """Get text embeddings to use with VLM"""
            text_inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=150
            ).to(device)
            return self.vlm.get_input_embeddings()(text_inputs["input_ids"])
