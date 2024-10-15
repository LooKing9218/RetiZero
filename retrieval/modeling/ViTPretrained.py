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
from retrieval.modeling import timm_models as timm

def ViT_Pretained(pretrained=True):
    model = timm.create_model('vit_large_patch32_224',num_classes=10,pretrained=pretrained)
    model.head = Identity()
    return model



from retrieval.modeling.LORA.lora_image_encoder import LoRA_ViT
def Lora_ViT_ImageNet(pretrained=True):
    print("================ Lora_ViT_ImageNet ================")
    ViT= ViT_Pretained(pretrained=pretrained)
    model = LoRA_ViT(ViT, r=4)
    return model
if __name__ == "__main__":
    net = ViT_Pretained(pretrained=False)
    # print("num_features == {}".format(num_features))
    # print("num_features_l == {}".format(num_features_l))
    images = torch.rand(2, 3, 224, 224)
    output = net(images)
    print(output.shape)