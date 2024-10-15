import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
from transformers import AutoImageProcessor
import tqdm
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from retrieval import CLIPRModel
import seaborn as sns
import umap.umap_ as umap
import torch, hdbscan
import pandas as pd

Data_Pkl = "ImageRetrieval"
data_dir = "./"  # 替换为您的数据集目录
# dataset structure
# data_dir
# ---class_dir1
# ---class_dir2
# ......
# ---class_dirN

weight_path = "./Pretrained/RetiZero.pth"

def precision_at_k(sm, query, k):
    true_category = DATA.loc[query]["category"]
    recommended = DATA["category"].loc[get_top_k(sm, query, k)]
    return len(recommended[recommended == true_category]) / k


def find_similar(sm, query):
    top_20 = reversed(np.argsort(sm[query])[-21:])[1:]  # Search top 20
    data = DATA[["image", "category"]].loc[top_20]

    similarity_scores = sm[query][top_20] * 100

    plt.clf()
    _, ax = plt.subplots(nrows=5, ncols=5, figsize=(15, 18))
    for i in range(5): ax[0, i].axis('off')

    ax[0, 2].imshow(DATA.loc[query]["image"])
    ax[0, 2].set_title("Query Image: {}".format(DATA.loc[query]["category"]))

    for idx in range(len(data)):
        sub_ax = ax[int(np.floor(idx / 5)) + 1][idx % 5]
        sub_ax.axis('off')

        if idx < len(data):
            row = data.iloc[idx]
            sub_ax.imshow(row["image"])
            cat, sim = row['category'], similarity_scores[idx]
            sub_ax.set_title(f"True category: {cat}; Sim: {sim:0.1f}%", fontsize=10)
    plt.tight_layout()

    plt.savefig("{}/find_similar_{}.png".format(ResultsPath,query))
#


class Model_Finetuing(torch.nn.Module):
    def __init__(self,model_name,weight_path):
        super().__init__()

        FLAIRModel_trained = CLIPRModel(vision_type=model_name, from_checkpoint=True,
                           weights_path=weight_path,R=8)
        self.img_encoder = FLAIRModel_trained.vision_model.model
        for para in self.img_encoder.parameters():
            para.requires_grad = False

    def forward(self,x):
        x_features = self.img_encoder(x)
        return x_features

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
to_pil = transforms.ToPILImage()


dataset = ImageFolder(root=data_dir, transform=transform)

ALL_IMAGES = []

for im_id in range(len(dataset)):
    image, label = dataset[im_id]
    filename = os.path.basename(dataset.imgs[im_id][0])
    pil_image = to_pil(image)

    # Convert PIL image to NumPy array
    np_image = np.array(pil_image)

    image_data = {
        "image": np_image,
        "category": dataset.classes[label],
        "filename": filename
    }
    ALL_IMAGES.append(image_data)


ALL_IMAGES = pd.DataFrame(ALL_IMAGES)
ALL_IMAGES.head(10)

ALL_IMAGES.to_pickle('all_images_backup.pkl')



ALL_IMAGES = pd.read_pickle('all_images_backup.pkl')

ALL_IMAGES.head(10)

import matplotlib.pyplot as plt

# 按类别分组
grouped_images = ALL_IMAGES.groupby('category')


ALL_IMAGES = ALL_IMAGES.groupby(
    by='category',
    as_index=False
).head(8000).sort_values("category")


embeddings_model = Model_Finetuing(model_name="lora", weight_path=weight_path).to("cuda")

embeddings_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

all_attentions = []
all_embeddings = []
all_captions = []

last_hidden_shape = None
attentions_shape = None
embeddings_shape = None

for start in tqdm.tqdm(range(0, len(ALL_IMAGES), 8)):
    ims = list(ALL_IMAGES["image"][start:start + 8])
    # print("ims =========== {}".format(ims))
    # Generate the embeddings
    embed_inputs = embeddings_processor(ims, return_tensors="pt").to("cuda")

    embed_last_hidden = embeddings_model(embed_inputs["pixel_values"])


    if last_hidden_shape is None:
        last_hidden_shape = embed_last_hidden.shape

    # Get the weights of each attention window in the last layer
    attentions = embed_last_hidden.norm(dim=2)
    all_attentions += attentions.tolist()

    if attentions_shape is None:
        attentions_shape = attentions.shape

    # Get the embeddings
    embeds = embed_last_hidden.mean(axis=1)
    all_embeddings += embeds.tolist()

    if embeddings_shape is None:
        embeddings_shape = embeds.shape

    # The input and outputs of the models take a lot of memory,
    # so we delete them here since we are not using them again
    del embed_last_hidden, embeds, attentions

    torch.cuda.empty_cache()

# Normalize the attention weights between 0 and 1
all_attentions = torch.tensor(all_attentions)
mn = all_attentions.min(1, keepdim=True)[0]
mx = all_attentions.max(1, keepdim=True)[0]

# Append the attentions, embeddings, and captions to the data frame
ALL_IMAGES["attention_weights"] = all_attentions.sub(mn).div(mx).tolist()
ALL_IMAGES["embedding"] = all_embeddings

# Delete unneeded variables
del all_attentions, all_embeddings, all_captions, mn, mx

# ALL_IMAGES.head(10)
ALL_IMAGES.to_pickle(Data_Pkl+".pkl")


ResultsPath = "ResultsPath"
if not os.path.exists(ResultsPath):
    os.makedirs(ResultsPath)

DATA = pd.read_pickle(Data_Pkl+".pkl").reset_index(drop=True)
# 按类别分组
grouped_images = DATA.groupby('category')

# 遍历每个类别

embeds = torch.tensor(list(DATA["embedding"]))
scale = torch.norm(embeds, dim=1) / 2
SIMILARITY_MATRIX = scale / (scale + torch.cdist(embeds, embeds))

del embeds, scale

print("Is symmetric:", (torch.isclose(SIMILARITY_MATRIX, SIMILARITY_MATRIX.T, 0.000001)).all())
sns.set_style("darkgrid")


get_top_k = lambda sm, query, k: reversed(np.argsort(sm[query])[-(k + 1):])[1:]



embeddings_256d = umap.UMAP(
    n_neighbors=20,
    n_components=256,
    min_dist=0.0,
    metric="euclidean",
    random_state=42,
).fit_transform(list(DATA["embedding"]))

#
clusters = hdbscan.HDBSCAN(
    min_cluster_size=100,
    metric='euclidean',
    cluster_selection_method='leaf',
).fit(embeddings_256d)



np.unique(clusters.labels_)
DATA["cluster"] = clusters.labels_

del clusters, embeddings_256d



embeddings_2d = umap.UMAP(
    n_neighbors=100,
    n_components=2,
    min_dist=0.0,
    metric="euclidean",
    random_state=42,
).fit_transform(list(DATA["embedding"]))



WEIGHTED_SIMILARITY_MATRIX = SIMILARITY_MATRIX.clone()

for c in DATA["cluster"].unique():
    cond = torch.from_numpy((DATA["cluster"] == c).to_numpy())
    mask = torch.where(cond.repeat(sum(cond), 1), 1, 0.5)
    WEIGHTED_SIMILARITY_MATRIX[cond] *= mask

find_similar(WEIGHTED_SIMILARITY_MATRIX, 128)
find_similar(WEIGHTED_SIMILARITY_MATRIX, 256)
find_similar(WEIGHTED_SIMILARITY_MATRIX, 512)


cw_precisions = np.array([precision_at_k(WEIGHTED_SIMILARITY_MATRIX, q, 1) for q in range(len(DATA))])
print("Overall Precision@1:", cw_precisions.mean())

cw_precisions = np.array([precision_at_k(WEIGHTED_SIMILARITY_MATRIX, q, 3) for q in range(len(DATA))])
print("Overall Precision@3:", cw_precisions.mean())

cw_precisions = np.array([precision_at_k(WEIGHTED_SIMILARITY_MATRIX, q, 5) for q in range(len(DATA))])
print("Overall Precision@5:", cw_precisions.mean())