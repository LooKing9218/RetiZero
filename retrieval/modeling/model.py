import torch
import os

from .dictionary import definitions
from . import constants
from .misc import wget_gdrive_secure

from torch.cuda.amp import autocast
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, logging
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CLIPRModel(torch.nn.Module):
    def __init__(self, vision_type='resnet_v1', bert_type='emilyalsentzer/Bio_ClinicalBERT', vision_pretrained=True,
                 proj_dim=512, proj_bias=False, logit_scale_init_value=0.07, from_checkpoint=True, weights_path=None,
                 out_path=None, image_size=512, caption="A fundus photograph of [CLS]", projection=True,
                 norm_features=True,R=8):
        super().__init__()

        # Set attributes
        self.vision_type = vision_type
        self.bert_type = bert_type
        self.vision_pretrained = vision_pretrained
        self.proj_dim = proj_dim
        self.proj_bias = proj_bias
        self.logit_scale_init_value = logit_scale_init_value
        self.from_checkpoint = from_checkpoint
        self.weights_path = weights_path
        self.out_path = out_path
        self.image_size = image_size
        self.caption = caption
        # Use of projection head and feature normalization on visione encoder
        # (only relevant during transferability stage)
        self.projection = projection
        self.norm_features = norm_features

        # Set vision and text encoder
        self.vision_model = VisionModel(vision_type=self.vision_type, pretrained=self.vision_pretrained,
                                        proj_dim=self.proj_dim, proj_bias=self.proj_bias, projection=self.projection,
                                        norm=self.norm_features,R=R)
        self.text_model = TextModel(bert_type=self.bert_type, proj_dim=self.proj_dim, proj_bias=self.proj_bias,
                                    projection=self.projection, norm=self.norm_features)

        # learnable temperature for contrastive loss
        self.logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1/self.logit_scale_init_value)))

        # Load pretrained weights
        if from_checkpoint:
            self.load_from_pretrained(self.weights_path)
        #
        # Set model to device
        self.to(device)

    def load_from_pretrained(self, weights_path=None):
        state_dict = torch.load(weights_path)
        self.load_state_dict(state_dict, strict=True)
        print('load model weight from:', weights_path)

    def softce_clip_loss(self, logits_per_text, target_pseudo):
        caption_loss = self.ce_loss(logits_per_text, target_pseudo)
        image_loss = self.ce_loss(logits_per_text.T, target_pseudo)
        return (caption_loss + image_loss) / 2.0

    def ce_loss(self, pred_logit, ref):
        ce_loss = torch.nn.functional.cross_entropy(pred_logit, ref)
        return ce_loss

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def fit(self, datalaoders, epochs=30, lr=5e-4, weight_decay=1e-5, scheduler=True, warmup_epoch=1, store_num=5,
            transforms=None):

        # Set optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        # Set scheduler
        if scheduler:
            from retrieval.pretraining.utils import get_scheduler_per_iteration
            scheduler = get_scheduler_per_iteration(optimizer, lr, warmup_epoch, len(datalaoders["train"]))
        else:
            scheduler = None

        # Training along epochs
        epoch = 1
        while epoch <= epochs:

            # Train epoch
            loss_epoch = self.train_epoch(datalaoders["train"], optimizer, scheduler, transforms, epoch)

            # Display epoch-wise loss
            print('Epoch=%d: ave_loss=%2.5f' % (epoch, loss_epoch))

            # Save model
            if epoch % store_num == 0:
                if self.out_path is not None:
                    if not os.path.isdir(self.out_path):
                        os.makedirs(self.out_path)
                    torch.save(self.state_dict(), self.out_path + self.vision_type + '_epoch' + str(epoch) + '.pth')

            # Update epoch
            epoch += 1

    def train_epoch(self, loader, optimizer, scheduler=None, transforms=None, epoch=1):
        self.train()
        max_grad_norm, scaler = 1, torch.cuda.amp.GradScaler()
        loss_ave = 0.0

        # Set iterator
        epoch_iterator = tqdm(
            loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        # Iterate trough training batches
        for step, batch in enumerate(epoch_iterator):
            # Retrieve documents
            images = batch["image"].to(device).to(torch.float32)
            # Create text tokens
            # print("list(batch[report][0] ======== {}".format(list(batch["report"][0])))
            text_tokens = self.text_model.tokenize(batch["report"])
            input_ids = text_tokens["input_ids"].to(device).to(torch.long)
            attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)

            # Create similarity matrix with soft labels as ground truth
            # coocurrence = np.array(
            #     [[iDesc == iiDesc for iDesc in batch["sel_category"]] for iiDesc in batch["sel_category"]], np.float32)
            # target = torch.tensor(coocurrence / coocurrence.sum(-1)).to(device).to(torch.float32)
            # labels = Variable(torch.LongTensor(range(batch_size))).to(cnn_code.device)
            batch_size = images.shape[0]
            # print("batch_size ===== {}".format(batch_size))
            target = torch.LongTensor(range(batch_size)).to(device)#.to(torch.float32)
            # Forward
            with autocast():

                # Image augmentation
                if transforms is not None:
                    images = transforms(images)

                # Forward vision and text encoder
                img_embeds = self.vision_model(images)
                # print("img_embeds.shape ===== {}".format(img_embeds.shape))
                text_embeds = self.text_model(input_ids, attention_mask)
                # print("text_embeds.shape ===== {}".format(text_embeds.shape))

                # Compute similarity matrix and logits
                logits_per_image = self.compute_logits(img_embeds, text_embeds)
                # print("logits_per_image.shape ===== {}".format(logits_per_image.shape))
                logits_per_text = logits_per_image.t()
                # print("logits_per_text.shape ===== {}".format(logits_per_text.shape))

                # Compute cross-entropy loss
                loss = self.softce_clip_loss(logits_per_text, target)

            # Update model with scaler
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Overall losses track
            loss_ave += loss.item()
            torch.cuda.empty_cache()

            # Set description
            epoch_iterator.set_description(
                "Epoch=%d: Training (%d / %d Steps) " % (epoch, step + 1, len(loader)) +
                "- loss_value: " + str(round(loss.item(), 3))
            )

            # Update optimizer scheduler
            if scheduler is not None:
                scheduler.step()

        self.eval()
        return loss_ave / len(loader)

    def forward(self, image, text):
        self.eval()

        # Pre-process image
        image = self.preprocess_image(image)

        # Pre-process text
        text_input_ids, text_attention_mask = self.preprocess_text(text)

        # Forward vision and text encoder
        with torch.no_grad():
            img_embeds = self.vision_model(image)
            text_embeds = self.text_model(text_input_ids, text_attention_mask)

            # Compute similarity matrix and logits
            logits = self.compute_logits(img_embeds, text_embeds)

            # Compute probabilities
            probs = logits.softmax(dim=-1)

        return probs.cpu().numpy(), logits.cpu().numpy()

    def preprocess_image(self, image):
        import torchvision.transforms as transforms
        from torchvision.transforms import Compose
        transforms_proce = Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            # CopyDict(),
            # LoadImage(),
            # ImageScaling(),
            # ProduceDescription(caption=caption),
            # AugmentDescription(augment=augment_description),
            # SelectRelevantKeys()
        ])

        img = transforms_proce(image)
        # print("img.shape ===== {}".format(img.shape))

        # Set format and device
        # image = image.to(torch.float32).to(device)
        img = torch.unsqueeze(img,dim=0).to(device)
        # print("img.shape ===== {}".format(img.shape))

        return img


    def preprocess_text(self, text):

        # Create text prompt
        prompts = [self.caption.replace("[CLS]", category) for category in text]

        # Create text tokens
        text_tokens = self.text_model.tokenize(prompts)
        input_ids = text_tokens["input_ids"].to(device).to(torch.long)
        attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)

        return input_ids, attention_mask

    def compute_text_embeddings(self, categories, domain_knowledge=False):
        # Obtain text embeddings per class
        text_embeds_dict = {}
        for iKey in range(len(categories)):

            # Replace text prompt with expert knowledge descriptions
            if domain_knowledge and categories[iKey] in list(definitions.keys()):
                descriptions = definitions[categories[iKey]]
                if categories[iKey] not in descriptions:
                    descriptions.append(categories[iKey])
            else:
                descriptions = [categories[iKey]]

            # Forwards prompts trough text encoder
            with torch.no_grad():
                print(descriptions)
                descriptions = [self.caption.replace("[CLS]", iDescription) for iDescription in descriptions]
                text_token = self.text_model.tokenizer(descriptions, truncation=True, padding=True, return_tensors='pt')
                input_ids = text_token["input_ids"].to(device).to(torch.long)
                attention_mask = text_token["attention_mask"].to(device).to(torch.long)

                text_embeds = self.text_model(input_ids, attention_mask)

            text_embeds_dict[categories[iKey]] = text_embeds.mean(0).unsqueeze(0)

        text_embeds_dict = text_embeds_dict
        text_embeds = torch.concat(list(text_embeds_dict.values()))

        return text_embeds_dict, text_embeds


class VisionModel(torch.nn.Module):
    def __init__(self, vision_type='resnet', pretrained=True, proj_dim=512, proj_bias=False, projection=True,
                 norm=True,R=8):
        super().__init__()
        self.proj_dim = proj_dim

        from retrieval.modeling.LoraRETFound import lora
        self.model = lora(pretrained=True,R=R)
        self.vision_dim = 1024

        # Set output dimension
        if projection:
            self.out_dim = self.proj_dim
        else:
            self.out_dim = self.vision_dim

        # Set projection head
        self.projection_head_vision = ProjectionLayer(layer=torch.nn.Linear(self.vision_dim, self.proj_dim,
                                                                            bias=proj_bias),
                                                      projection=projection, norm=norm)

    def forward(self, pixel_values):
        embed = self.model(pixel_values)
        return embed


class TextModel(torch.nn.Module):
    def __init__(self, bert_type='emilyalsentzer/Bio_ClinicalBERT', proj_dim=512, proj_bias=False, projection=True,
                 norm=True):
        super().__init__()

        # Set tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.tokenizer.model_max_length = 77

        # Load text encoder from pretrained
        self.model = AutoModel.from_pretrained(bert_type, output_hidden_states=True)

        # Set projection head
        self.projection_head_text = ProjectionLayer(layer=torch.nn.Linear(768, proj_dim, bias=proj_bias),
                                                    projection=projection, norm=norm)

    def tokenize(self, prompts_list):
        text_tokens = self.tokenizer(prompts_list, truncation=True, padding=True, return_tensors='pt')
        return text_tokens

    def forward(self, input_ids, attention_mask):

        # Forwards trough text encoder
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Combine last feature layers to compute text embedding
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2],
                                          output['hidden_states'][-1]])
        embed = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1)

        # Compute projection from text embedding to multi-modal projection
        embed = self.projection_head_text(embed)
        return embed


class ProjectionLayer(torch.nn.Module):
    def __init__(self, layer, projection=True, norm=True):
        super().__init__()

        self.apply_projection = projection
        self.norm_modality = bool(projection * norm)
        self.norm_projection = norm
        self.projection = layer

    def forward(self, x):

        if self.norm_modality:
            x = x / x.norm(dim=-1, keepdim=True)

        if self.apply_projection:
            x = self.projection(x)
            if self.norm_projection:
                x = x / x.norm(dim=-1, keepdim=True)

        return x