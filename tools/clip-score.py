import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np


def load_clip_model_and_processor():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


def calculate_clip_score(image_path, text_path, model, processor):
    # Load image and text
    image = Image.open(image_path).convert("RGB")
    with open(text_path, 'r') as file:
        text = file.read().strip()

    # Process inputs
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)

    # Calculate cosine similarity
    image_features = outputs.image_embeds
    text_features = outputs.text_embeds
    similarity = torch.matmul(image_features, text_features.T).squeeze().item()

    return similarity


def match_and_score_images_texts(image_folder, text_folder, model, processor):
    image_files = {os.path.splitext(f)[0]: os.path.join(image_folder, f) for f in os.listdir(image_folder) if
                   f.lower().endswith('.png')}
    text_files = {os.path.splitext(f)[0]: os.path.join(text_folder, f) for f in os.listdir(text_folder) if
                  f.lower().endswith('.txt')}

    matched_scores = []

    for name, image_path in image_files.items():
        if name in text_files:
            text_path = text_files[name]
            score = calculate_clip_score(image_path, text_path, model, processor)
            matched_scores.append((name, score))

    return matched_scores


def main(image_folder, text_folder):
    model, processor = load_clip_model_and_processor()
    matched_scores = match_and_score_images_texts(image_folder, text_folder, model, processor)

    for name, score in matched_scores:
        print(f"File name: {name}, CLIP Score: {score:.4f}")


if __name__ == "__main__":
    image_folder = "./image"
    text_folder = "./text"
    main(image_folder, text_folder)