## Execute this cell to load all the requirements

# Image libraries
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
from scipy import ndimage

# Progress bar
from tqdm import tqdm

# DeepLearning library
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(0)


def generate_random_square(idx, img_size, n_rectangle):
  # Set seed
  np.random.seed(idx+1)

  # Number of random rectangles to place in the image
  num_rectangles = np.random.randint(10, n_rectangle)
  gray_scale_values = np.random.randint(0,255, size=num_rectangles+1)
  random_positions = np.random.randint(0, img_size-1, size=(num_rectangles, 2))

  # Generate random grayscale levels for the background and rectangles
  image = np.ones(shape=(img_size, img_size)) * np.random.randint(0,256)

  for i in range(num_rectangles):
    # Random rectangle size and position
    x1, y1 = random_positions[i, :]
    x2, y2 = np.random.randint(x1 + 1, img_size), np.random.randint(y1 + 1, img_size)

    # Draw the rectangle on the image
    image[y1:y2, x1:x2] = gray_scale_values[i]

  return image


def generate_label(image, mode, img_size, threshold):
  # Create an edge map with the same size as the image
  sobel_h = np.zeros((img_size, img_size))
  sobel_v = np.zeros((img_size, img_size))

  # Detect horizontal and vertical edges
  if mode in ['horizontal', 'both']:
    sobel_h = ndimage.sobel(image, 0)  # horizontal gradient
  if mode in ['vertical', 'both']:
    sobel_v = ndimage.sobel(image, 1)  # vertical gradient

  # Compute edge intensities
  magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
  magnitude *= 255.0 / np.max(magnitude)  # normalization

  edge_label = np.uint8(magnitude > threshold)

  return edge_label


def add_gaussian_noise(image, mean=0, std=10):
    """
    Add Gaussian noise to an image.
    Args:
        image (numpy array): The input image (before normalization).
        mean (float): The mean of the Gaussian distribution.
        std (float): The standard deviation of the Gaussian distribution.
    """
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    # Clip to maintain valid pixel range
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image


class RandomRectangleDataset(Dataset):
  def __init__(self, mode='both', N_sample=1024, img_size=224, threshold=8, n_rectangle=25):
    """
    Args:
        mode (str): Type of edges to detect ('horizontal', 'vertical, or 'both').
        num_samples (int): Number of samples in the dataset.
        image_size (int): Size of the square image (image_size x image_size).
        threshold (int): Grayscale difference threshold for labeling an edge.
        n_rectangle (int): Maximal number of rectangle to draw in an image.
    """
    self.mode = mode
    self.N_sample = N_sample
    self.img_size = img_size
    self.threshold = threshold
    self.n_rectangle = n_rectangle

  def __len__(self):
    return self.N_sample

  def __getitem__(self, idx):
    # Call generate_random_square function
    image = generate_random_square(idx, self.img_size, self.n_rectangle)

    # Call generate_label function
    label = generate_label(image, self.mode, self.img_size, self.threshold)

    # Convert label (224,224) uint label to a (1,224,224)-float32 tensor
    label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

    # Add random noise
    noised_image = add_gaussian_noise(image, std=np.random.uniform(1,8))

    # Normalize to [0,1] and switch to float32 tensor
    image_tensor = torch.tensor(noised_image, dtype=torch.float32).unsqueeze(0) / 255.0

    return image_tensor, label_tensor