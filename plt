import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image

# ========================
# 1️⃣ Confusion Matrix - Load image
# ========================
cm_2y_img = Image.open("plots/cm_2y_ensemble.png")
cm_3y_img = Image.open("plots/cm_3y_ensemble.png")

plt.figure(figsize=(8,6))
plt.imshow(cm_2y_img)
plt.axis('off')
plt.title("Confusion Matrix - 2-Year Ensemble")
plt.show()

plt.figure(figsize=(8,6))
plt.imshow(cm_3y_img)
plt.axis('off')
plt.title("Confusion Matrix - 3-Year Ensemble")
plt.show()

# ========================
# 2️⃣ Attention Map - Load image
# ========================
attn_2y_img = Image.open("plots/attention_2y.png")
attn_3y_img = Image.open("plots/attention_3y.png")

plt.figure(figsize=(10,6))
plt.imshow(attn_2y_img)
plt.axis('off')
plt.title("Attention Map - 2-Year Ensemble")
plt.show()

plt.figure(figsize=(10,6))
plt.imshow(attn_3y_img)
plt.axis('off')
plt.title("Attention Map - 3-Year Ensemble")
plt.show()
