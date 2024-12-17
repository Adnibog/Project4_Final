import os
import cv2
import numpy as np

def load_images_from_folder(folder):
    images = []
    labels = []
    label_map = {}
    current_label = 0

    for subject in os.listdir(folder):
        subject_folder = os.path.join(folder, subject)
        if os.path.isdir(subject_folder):
            if subject not in label_map:
                label_map[subject] = current_label
                current_label += 1
            for filename in os.listdir(subject_folder):
                img_path = os.path.join(subject_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(label_map[subject])
    return np.array(images), np.array(labels)
