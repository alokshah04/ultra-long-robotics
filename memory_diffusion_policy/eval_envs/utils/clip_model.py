import torch
from transformers import CLIPTokenizer, CLIPConfig, CLIPModel
from huggingface_hub import try_to_load_from_cache, hf_hub_download
from safetensors.torch import load_file as _st_load_file


class ClipTextEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

        # Build model from config (JSON only, no torch.load)
        config = CLIPConfig.from_pretrained(model_name)
        self.model = CLIPModel(config)

        # Load weights while bypassing transformers' torch>=2.6 requirement
        # (CVE-2025-32434 guard added in transformers 4.47+).
        # Strategy: try safetensors → fall back to pytorch_model.bin via direct torch.load
        # try_to_load_from_cache returns a str path, None (not checked), or a
        # _CACHED_NO_EXIST sentinel — use isinstance(str) to distinguish.
        sf_path = try_to_load_from_cache(model_name, "model.safetensors")
        if isinstance(sf_path, str):
            self.model.load_state_dict(_st_load_file(sf_path))
        else:
            # pytorch_model.bin is safe to load directly with weights_only=True,
            # bypassing transformers' torch>=2.6 version guard entirely.
            bin_path = try_to_load_from_cache(model_name, "pytorch_model.bin")
            if not isinstance(bin_path, str):
                bin_path = hf_hub_download(model_name, "pytorch_model.bin")
            sd = torch.load(bin_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(sd, strict=False)

        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def embed_texts(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        feats = self.model.get_text_features(**inputs)  # [B, D]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().cpu()  # torch.Tensor on CPU
