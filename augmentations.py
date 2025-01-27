import numpy as np
import random

def random_augmentation(points, colors, use_color):
    augmentations = [
        scale_object,
        rotate_object,
        translate_object,
        jitter_object,
        identity,
        #add_rgb_noise_to_object,
        #adjust_brightness_of_object,
        #adjust_contrast_of_object
    ]
    
    if use_color:
        augmentations.append(add_rgb_noise_to_object)
        augmentations.append(adjust_brightness_of_object)
        augmentations.append(adjust_contrast_of_object)
    
    # Select a random augmentation
    augmentation = random.choice(augmentations)
    
    # Apply the selected augmentation
    return augmentation(points, colors)


def identity(points, colors):
    return points, colors


def scale_object(points, colors, scale_range=(0.9, 1.1)):
    # Apply scaling only to the object points
    scale = np.random.uniform(*scale_range)
    points *= scale
    return points, colors

def rotate_object(points, colors, angle_range=(0, 0.5)):
    # Apply rotation only to the object points
    angle = np.random.uniform(*angle_range)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    points = points @ rotation_matrix.T
    return points, colors


def translate_object(points, colors, translate_range=(-0.1, 0.1)):
    # Apply translation only to the object points
    translation = np.random.uniform(*translate_range, size=(1, 3))
    points += translation
    return points, colors


def jitter_object(points, colors, sigma=0.005):
    # Apply jitter only to the object points
    noise = np.random.normal(0, sigma, points.shape)
    points += noise
    return points, colors


def add_rgb_noise_to_object(points, colors, sigma=0.05):
    # Add noise to the RGB values only
    noise = np.random.normal(0, sigma, colors.shape)
    colors += noise
    colors = np.clip(colors, -1, 1)  # Ensure RGB values stay within [-1, 1]
    return points, colors


def adjust_brightness_of_object(points, colors, factor_range=(0.7, 1.3)):
    # Adjust brightness of the object's RGB values
    factor = np.random.uniform(*factor_range)
    colors *= factor
    colors = np.clip(colors, -1, 1)  # Ensure RGB values stay within [-1, 1]
    return points, colors


def adjust_contrast_of_object(points, colors, factor_range=(0.7, 1.3)):
    # Adjust the contrast of the object's RGB values
    mean_rgb = np.mean(colors, axis=0)
    contrast_factor = np.random.uniform(*factor_range)
    colors = (colors - mean_rgb) * contrast_factor + mean_rgb
    colors = np.clip(colors, -1, 1)  # Ensure RGB values stay within [-1, 1]
    return points, colors

