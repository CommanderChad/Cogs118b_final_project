#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import requests
import torch
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import sys
print(sys.executable)  # just to see which Python the notebook is using

# Install into THIS Python, not some other one
get_ipython().system('{sys.executable} -m pip install --upgrade diffusers[torch] transformers')
from diffusers import StableUnCLIPImg2ImgPipeline
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


# ──────────────────────────────────────────────────────────────
# 1.  Load unCLIP – vision side only (projection_dim = 1024)   ─
# ──────────────────────────────────────────────────────────────
pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "sd2-community/stable-diffusion-2-1-unclip",
    torch_dtype=torch.float16,
    variant="fp16"
).to(device)

vision_encoder = pipe.image_encoder         


# In[4]:


openclip_repo = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"     # projection_dim = 1024 :contentReference[oaicite:0]{index=0}
tokenizer = CLIPTokenizer.from_pretrained(openclip_repo)
text_encoder = CLIPTextModelWithProjection.from_pretrained(
    openclip_repo,
    torch_dtype=torch.float16
).to(device)

# optional: stuff them into the pipe so `pipe.tokenizer` etc. work
pipe.tokenizer, pipe.text_encoder = tokenizer, text_encoder


# In[5]:


def embed_images(paths, batch_size=8):
    """Return (N,1024) image embeddings"""
    out, fe, enc = [], pipe.feature_extractor, pipe.image_encoder
    for i in range(0, len(paths), batch_size):
        print(f"Batch {i}/{len(paths)}")
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
        px   = fe(imgs, return_tensors="pt").pixel_values.to(enc.device, enc.dtype)
        with torch.no_grad():
            v = enc(px)[0]                              # (B,1024)
        out.append(v)
    return torch.cat(out)  # (N,1024)


# In[6]:


# ──────────────────────────────────────────────────────────────
#  Deal with file structure and get working paths
# ──────────────────────────────────────────────────────────────
root = "THINGS_animalgroups"

# collect all jpgs and keep their group (top-level folder)
groupedimages = {}

for group in os.listdir(root):
    group_dir = os.path.join(root, group)
    groupedimages[group] = []

    #go into animal name from category
    for animal in os.listdir(group_dir):
        animal_dir = os.path.join(group_dir, animal)
        if not os.path.isdir(animal_dir):
            continue

        # animal images inside animal files
        for fname in os.listdir(animal_dir):
            if fname.lower().endswith(".jpg"):
                full_path = os.path.join(animal_dir, fname)
                groupedimages[group].append(full_path)


# In[7]:


#map each group
groups = sorted(groupedimages.keys())   # deterministic order
group_to_idx = {g: i for i, g in enumerate(groups)}
print("Label mapping:", group_to_idx)

all_paths = []
all_labels = []

for group, paths in groupedimages.items():
    for p in paths:
        all_paths.append(p)
        all_labels.append(group_to_idx[group])

all_labels = torch.tensor(all_labels, dtype=torch.long)


# In[ ]:


#embedding using given image embedding
with torch.no_grad():
    img_feats = embed_images(all_paths) # (N, 1024)
    img_feats = img_feats.to(torch.float32)
    img_feats = F.normalize(img_feats, dim=-1)  


# In[9]:


img_np = img_feats.cpu().numpy()
labels_np = all_labels.cpu().numpy()

# 80/20 split
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    img_np, labels_np,
    test_size=0.20,
    random_state=42,
    stratify=labels_np
)
X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
X_test  = torch.tensor(X_test_np,  dtype=torch.float32).to(device)
y_train = torch.tensor(y_train_np, dtype=torch.long).to(device)
y_test  = torch.tensor(y_test_np, dtype=torch.long).to(device)


# In[14]:


num_classes = len(groups)
embed_dim = img_feats.shape[1]

print("num classes:", num_classes)
print("embedding dim:", embed_dim)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# basic vlassifier
classifier = nn.Linear(embed_dim, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
epochs = 100  

for epoch in range(epochs):
    classifier.train()
    total_loss = 0.0

    for feats_batch, labels_batch in train_loader:
        feats_batch = feats_batch.to(device)
        labels_batch = labels_batch.to(device)

        logits = classifier(feats_batch)
        loss = criterion(logits, labels_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels_batch.size(0)

    avg_loss = total_loss / len(train_dataset)

    #monitor training accuracy
    if (epoch + 1) % 50 == 0 or epoch == 1:
        classifier.eval()
        with torch.no_grad():
            logits_train = classifier(X_train)
            preds_train = logits_train.argmax(dim=-1).cpu()
            acc = (preds_train == y_train.cpu()).float().mean().item()
        print(f"epoch {epoch+1:3d}/{epochs} | loss = {avg_loss:.4f} | train acc = {acc:.3f}")


# In[15]:


classifier.eval()
with torch.no_grad():
    logits_test = classifier(X_test)
    preds_test = logits_test.argmax(dim=-1).cpu()
    test_acc = (preds_test == y_test.cpu()).float().mean().item()

print(f"\ntest accuracy = {test_acc:.3f}")


# In[16]:


# map class indices back to group names
idx_to_group = {i: g for g, i in group_to_idx.items()}
classificationResults = []

def classifyNewImage(img_path, topk=None):
    
    classifier.eval()
    with torch.no_grad():

        feat = embed_images([img_path])
        feat = feat.to(torch.float32)
        feat = F.normalize(feat, dim=-1)
        
        logits = classifier(feat.to(next(classifier.parameters()).device))
        probs = logits.softmax(dim=-1)[0].cpu().numpy()

    pred_idx = probs.argmax()
    pred_group = idx_to_group[pred_idx]
    pred_conf = float(probs[pred_idx])
    pred_conf_pct = pred_conf * 100.0
    
    #to create data table display
    row = {
        "image": os.path.basename(img_path),
        "path": img_path,
        "pred_group": pred_group,
        "confidence_pct": pred_conf_pct,
    }

    classificationResults.append(row)

    print(f"image: {img_path}")
    print(f"predicted group: {pred_group} ({pred_conf_pct:.1f}% confidence)")
    print("Class probabilities:")
    for i, g in enumerate(groups):
        print(f"  {g:>12}: {probs[i]*100:.1f}%")

    if topk is not None:
        # return top-k labels + probs as a small list for analysis
        topk_idx = probs.argsort()[::-1][:topk]
        topk_labels = [idx_to_group[i] for i in topk_idx]
        topk_probs = probs[topk_idx]
        return pred_group, probs, topk_labels, topk_probs

    return pred_group, probs


# In[17]:


cm = confusion_matrix(y_test.cpu().numpy(), preds_test.numpy())

fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=groups)
disp.plot(ax=ax, cmap="Blues", values_format="d")

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_title("Confusion Matrix (Test Data)")
plt.show()


# In[18]:


classifyNewImage('PCA_images/alligator_original.jpg')


# In[19]:


classifyNewImage('PCA_images/alligator_remade.jpg')


# In[20]:


classifyNewImage('PCA_images/beetle_original.jpg')


# In[21]:


classifyNewImage('PCA_images/beetle_remade1.jpg')


# In[24]:


classifyNewImage('PCA_images/beetle_remade2.jpg')


# In[25]:


classifyNewImage('PCA_images/elephant_original.jpg')


# In[26]:


classifyNewImage('PCA_images/elephant_remade.jpg')


# In[27]:


classifyNewImage('PCA_images/hawk_original.jpg')


# In[28]:


classifyNewImage('PCA_images/hawk_remade.jpg')


# In[29]:


classifyNewImage('PCA_images/bee_original.jpg')


# In[30]:


classifyNewImage('PCA_images/bee_remade.jpg')


# In[34]:


classifyNewImage('PCA_images/shark_original.jpg')


# In[35]:


classifyNewImage('PCA_images/shark_remade.jpg')


# In[33]:


results= pd.DataFrame(classificationResults)
results


# In[ ]:




