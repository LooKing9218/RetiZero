import torch
import torch.nn as nn
from retrieval.modeling.models_vit import VisionTransformer
from functools import partial
from retrieval.modeling.pos_embed import interpolate_pos_embed
from retrieval.modeling.timm_models.models.layers import trunc_normal_
class Identity(nn.Module):
    """Identity layer to replace last fully connected layer"""

    def forward(self, x):
        return x
def vit_large_patch16(pretrained=True):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    # manually initialize fc layer
    trunc_normal_(model.head.weight, std=2e-5)
    # print("Model = %s" % str(model))
    model.head = Identity()
    return model


from retrieval.modeling.LORA.lora_image_encoder import LoRA_ViT
def lora(pretrained=True,R=8):
    ViT= vit_large_patch16(pretrained=pretrained)
    model = LoRA_ViT(ViT, r=R)
    return model


if __name__ == "__main__":
    net = vit_large_patch16(pretrained=False)
    # print("num_features == {}".format(num_features))
    # print("num_features_l == {}".format(num_features_l))
    images = torch.rand(2, 3, 224, 224)
    output = net(images)
    print(output.shape)