import matplotlib.pyplot as plt
import pandas as pd

# Load the data
df = pd.read_csv('C:\\Users\\Alen\\Desktop\\CV\\Object detection\\YOLO\\runs\\segment\\train5\\results.csv')

# Plot the loss values over time
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss')
plt.plot(df['epoch'], df['train/seg_loss'], label='Seg Loss')
plt.plot(df['epoch'], df['train/cls_loss'], label='Cls Loss')
plt.plot(df['epoch'], df['train/dfl_loss'], label='Dfl Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.show()