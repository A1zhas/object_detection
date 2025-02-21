from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
import matplotlib.pyplot as plt
import cv2

# Load model and processor
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# Load the image
image_path = "C:/Users/Alen/Desktop/CV/Object_detection/ZERO_SHOT/images/4.jpg"
image = Image.open(image_path)

# Define the prompts
prompts = ["a bird", "wood", "a wing"]

# Preprocess the input (text and image)
inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")

# Predict the segmentation masks
with torch.no_grad():
    outputs = model(**inputs)

# Extract logits (predictions)
preds = outputs.logits.unsqueeze(1)

# Visualize the results
_, ax = plt.subplots(1, 4, figsize=(15, 4))
for a in ax.flatten():
    a.axis('off')
ax[0].imshow(image)

# Display each mask with its corresponding prompt
for i in range(3):
    ax[i+1].imshow(torch.sigmoid(preds[i][0]).cpu().numpy())
    ax[i+1].text(0, -15, prompts[i], fontsize=12, color='white')

# Processing the mask for a specific class (for example, class 1: "wood")
i = 1  # class index
gray_image = torch.sigmoid(preds[i][0]).cpu().numpy() * 255

# Convert to binary image using thresholding
thresh, bw_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

# Fix color format (ensure correct RGB order)
bw_image_rgb = cv2.cvtColor(bw_image.astype('uint8'), cv2.COLOR_BGR2RGB)

# Display the binary image
plt.imshow(bw_image_rgb)
plt.axis('off')
plt.show()
