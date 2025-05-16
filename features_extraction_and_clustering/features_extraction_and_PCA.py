import os
import json
import base64
from io import BytesIO
from PIL import Image
import torch
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from transformers import pipeline


import numpy as np

# The 28 emotions used by the GoEmotions model
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

# Pre-trained RoBERTa emotion classifier
emotion_classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

# For accelerated computation with PyTorch if possible
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP's model and processor
CLIPModel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
CLIPProcessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Define the name of the extracted text (in our case the same as for filtering)
extracted_text_json = "extracted_text_for_check.json"

def get_emotion_embedding_from_image(image):
    """
    Predicts the emotion and returns the emotion embedding for an image using CLIP.
    """
    # Preprocess the image and text prompts
    if isinstance(image, str):
        image = Image.open(image)
    inputs = CLIPProcessor(text=EMOTION_LABELS, images=image, return_tensors="pt", padding=True).to(device)

    # Get predictions
    with torch.no_grad():
        outputs = CLIPModel(**inputs)
        logits_per_image = outputs.logits_per_image  # Image-text similarity scores
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()  # Convert to probabilities

    # Get the predicted emotion
    predicted_emotion_index = np.argmax(probs)
    predicted_emotion = EMOTION_LABELS[predicted_emotion_index]

    # Create the emotion embedding vector
    emotion_embedding = probs.flatten().tolist()  # Flatten and convert to list

    # Return emotion and embedding
    return {
        "predicted_emotion": predicted_emotion,
        "emotion_embedding": emotion_embedding
    }

def get_emotion_embedding_from_text(text):
    """
    Predicts the emotion and returns the emotion embedding for an image using CLIP.
    """
    emotions = emotion_classifier(text)
    emotion_vector = [0] * 28  # Create a 28-dimensional empy vector to save the probabilities

    for emotion in emotions[0]:  # Process multiple emotions (the 0. index is to get the emotions)
        label = emotion["label"]
        score = emotion["score"]
        index = EMOTION_LABELS.index(label)  # Find label index
        emotion_vector[index] = score  # Assign probability score

    return np.array(emotion_vector)

def extract_text_from_image(image_name, extracted_text_json):
    """Find entry in JSON of extracted texts by name."""
    for entry in extracted_text_json:
        if entry["image"] == image_name:
            return entry["text"]
    return ""


def encode_image_to_base64(image_path):
    """Encodes an image to a base64 string."""
    image = Image.open(image_path).convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def extract_image_features(image_path):
    """Extracts visual features using CLIP."""
    image = Image.open(image_path).convert("RGB") # Open the image 
    inputs = CLIPProcessor(images=image, return_tensors="pt").to(device) # Preprocess the image

    with torch.no_grad():
        outputs = CLIPModel.get_image_features(**inputs) # Get image features

    return outputs.cpu().numpy().flatten()

def extract_text_features(text, chunk_size=50, overlap=25, merge_method="mean"):
    """Extracts CLIP text embeddings from any length by using overlapping chunks."""
    # Tokenize the text
    tokens = CLIPProcessor.tokenizer(text, return_tensors="pt", truncation=False, padding=False)
    input_ids = tokens["input_ids"].squeeze(0) 
    num_tokens = input_ids.shape[0]

    # If the text is short enough, process it in one go
    if num_tokens <= 77:
        inputs = CLIPProcessor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            outputs = CLIPModel.get_text_features(**inputs)
        return outputs.cpu().numpy().flatten()

    # If the text is too long, process it as overlapping chunks
    embeddings = []
    start_idx = 0
    while start_idx < num_tokens:
        chunk_ids = input_ids[start_idx: start_idx + chunk_size]
        chunk_text = CLIPProcessor.tokenizer.decode(chunk_ids)
        inputs = CLIPProcessor(text=[chunk_text], return_tensors="pt", padding=True, truncation=True, max_length=77).to(
            device)

        with torch.no_grad():
            chunk_embedding = CLIPModel.get_text_features(**inputs).cpu().numpy().flatten()

        embeddings.append(chunk_embedding)
        start_idx += chunk_size - overlap

    embeddings = np.array(embeddings)

    # Merge the embeddings based on the specified method, mean for the thesis
    if merge_method == "mean":
        return np.mean(embeddings, axis=0)
    elif merge_method == "max":
        return np.max(embeddings, axis=0)
    elif merge_method == "concat":
        return np.concatenate(embeddings, axis=0)

    raise ValueError("Invalid merge_method.")



def extract_embeddings(image_directory, output_json):
    """Extracts image and text features from all images in a directory and saves them to a JSON file."""

    image_data_with_emotion = []
    image_data_without_emotion = []
    all_fused_embeddings_with_emotion = []
    all_fused_embeddings_without_emotion = []

    with open(extracted_text_json, "r") as file:
        json_data = json.load(file)

    # For each image in the directory
    for filename in tqdm(os.listdir(image_directory)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')): # Image file types
            image_path = os.path.join(image_directory, filename)

            extracted_text = extract_text_from_image(filename, json_data) # get text from JSON

            image_base64 = encode_image_to_base64(image_path) # Encode image to base64

            img_features = extract_image_features(image_path) # Extract image features
            text_features = extract_text_features(extracted_text) # Extract text features

            # Get emotion embeddings
            text_emotions = get_emotion_embedding_from_text(extracted_text) # From text
            result = get_emotion_embedding_from_image(image_path) # From image
            image_emotions = np.array(result["emotion_embedding"]) # Get only the embedding

            # Fuse embeddings and normalize
            scaler = StandardScaler()  # Z-score normalization 

            # Normalize all embeddings
            text_emb_norm = scaler.fit_transform(text_features.reshape(-1, 1)).flatten()
            visual_emb_norm = scaler.fit_transform(img_features.reshape(-1, 1)).flatten()
            emotion_t_emb_norm = scaler.fit_transform(text_emotions.reshape(-1, 1)).flatten()
            emotion_v_emb_norm = scaler.fit_transform(image_emotions.reshape(-1, 1)).flatten()

            # Concatenate normalized embeddings for both cases
            fused_embedding_with_emotion = np.concatenate([text_emb_norm, visual_emb_norm,
                                              emotion_t_emb_norm, emotion_v_emb_norm], axis=0)

            fused_embedding_without_emotion = np.concatenate([text_emb_norm, visual_emb_norm], axis=0)

            # Store data before PCA
            all_fused_embeddings_with_emotion.append(fused_embedding_with_emotion)
            all_fused_embeddings_without_emotion.append(fused_embedding_without_emotion)

            image_data_with_emotion.append({
                "image_name": filename,
                "image_base64": image_base64,
                "extracted_text": extracted_text,
                "fused_features": fused_embedding_with_emotion.tolist(),
            })

            image_data_without_emotion.append({
                "image_name": filename,
                "image_base64": image_base64,
                "extracted_text": extracted_text,
                "fused_features": fused_embedding_without_emotion.tolist(),
            })
    print("Embeddings retrieved/")
    print("Doing PCA.")
    pca_dim = 128 # Is optimal number of dimensions

    # Find the minimum number of components for PCA based on data (in thesis 128 because of the large dataset)
    all_fused_embeddings_with_emotion = np.array(all_fused_embeddings_with_emotion)
    n_components = min(pca_dim, all_fused_embeddings_with_emotion.shape[0], all_fused_embeddings_with_emotion.shape[1])

    # Perform PCA for case with emotion
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(all_fused_embeddings_with_emotion)

    # Store PCA results
    for i, data in enumerate(image_data_with_emotion):
        data["pca_features"] = reduced_embeddings[i].tolist()

    with open(output_json+"_with_emotion.json", "w") as f:
        json.dump(image_data_with_emotion, f, indent=4)

    # Perform PCA for case without emotion
    all_fused_embeddings_without_emotion = np.array(all_fused_embeddings_without_emotion)
    n_components = min(pca_dim, all_fused_embeddings_without_emotion.shape[0], all_fused_embeddings_without_emotion.shape[1])

    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(all_fused_embeddings_without_emotion)

    # Store PCA results
    for i, data in enumerate(image_data_without_emotion):
        data["pca_features"] = reduced_embeddings[i].tolist()

    with open(output_json+"_without_emotion.json", "w") as f:
        json.dump(image_data_without_emotion, f, indent=4)

    print("PCA done. Program finished.")
    return


# Entry point
if __name__ == "__main__":
    image_directory = "you/dataset/folder/memes_dataset"
    output_json = "embeddings_extraction" # part of the name of the output JSON file
    extract_embeddings(image_directory, output_json)
