import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
from PIL import Image
from zeroshot import CLIPRModel
# Set model
weight_path = "Pathto/RetiZero.pth"
Model_Pretrained = CLIPRModel(vision_type="lora",
                                from_checkpoint=False,
                                weights_path=weight_path, R=8)
state_dict = torch.load(weight_path)
Model_Pretrained.load_state_dict(state_dict,strict=True)
print("Weight load succesfull!!")


image = Image.open("Glaucoma/91085_Fundus.png").convert("RGB")
text = [
    "Normal",
    "Retinal Vein Occlusion",
    "Central Serous Chorioretinopathy",
    "Non-proliferative Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy",
    "Epiretinal Membrane",
    "Glaucoma",
    "Macular Hole",
    "Pathologic Maculopathy",
    "Retinal Artery Occulusion",
    "Retinal Detachment",
    "Retinitis Pigmentosa",
    "Vogt-Koyanagi-Harada (VKH) disease",
    "Age-related Macular Degeneration"
]


Probability, logits = Model_Pretrained(image, text)
pred = Probability.argmax(-1)
print("Prediction: {}, Probability: {}".format(
    text[pred],Probability[pred]
))