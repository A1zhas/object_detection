from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import numpy as np
import torch

# Load the model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load the image
image_path = "C:/Users/Alen/Desktop/CV/Object_detection/ZERO_SHOT/images/4.jpg"  # Используйте прямые слэши
image = Image.open(image_path)

# Display the image to verify it's loaded
image.show()  # Это откроет изображение в стандартном просмотрщике изображений

# Define text inputs
text_inputs = ["a photo of a red bird", "a photo of a blue bird", "a photo of dancing people"]

# Prepare the inputs for the model
inputs = processor(text=text_inputs, images=image, return_tensors="pt", padding=True)

# Get model outputs
outputs = model(**inputs)

# Image-text similarity scores
logits_per_image = outputs.logits_per_image

# Apply softmax to get probabilities
probs = logits_per_image.softmax(dim=1)

# Convert to numpy array for further use
probs = probs.detach().cpu().numpy()

# Print the probabilities
print(probs)

# Plot the confidence scores as a bar plot
plt.figure(figsize=(10, 6))
plt.bar(range(len(text_inputs)), probs[0], color='skyblue')  # Use range(len(text_inputs)) for the x-axis positions
plt.xticks(range(len(text_inputs)), text_inputs, rotation=45, ha='right')  # Rotate x-axis labels
plt.xlabel('Text Inputs')
plt.ylabel('Confidence')
plt.title('Confidence Scores for Text Inputs')
plt.tight_layout()

# Show the plot and wait for it to close
plt.show()
