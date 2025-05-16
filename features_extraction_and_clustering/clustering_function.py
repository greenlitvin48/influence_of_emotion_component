import json
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity


def cluster_memes(input_json, output_json):
    # Load the JSON file
    with open(input_json, "r") as f:
        image_data = json.load(f)

    # Extract the features
    features_matrix = np.array([data["pca_features"] for data in image_data])

    # Perform clustering
    clustering_model = AffinityPropagation()
    clustering_model.fit(features_matrix)
    cluster_labels = clustering_model.labels_

    # Assign cluster IDs
    for i, data in enumerate(image_data):
        data["cluster_id"] = int(cluster_labels[i])

    # Compute inter-cluster similarities
    unique_clusters = np.unique(cluster_labels)
    cluster_centers = clustering_model.cluster_centers_
    inter_cluster_similarities = cosine_similarity(cluster_centers)

    # Normalize similarities to weights between 0 and 1
    inter_cluster_weights = {}
    for i, cluster_id in enumerate(unique_clusters):
        inter_cluster_weights[int(cluster_id)] = {}
        for j, other_cluster_id in enumerate(unique_clusters):
            if i != j:
                weight = inter_cluster_similarities[i, j]
                inter_cluster_weights[int(cluster_id)][int(other_cluster_id)] = weight

    # Save results to JSON
    output_data = {
        "images": [
            {"image_name": data["image_name"], "pca_features": data["pca_features"],
             "cluster_id": data["cluster_id"], "image_base64": data["image_base64"]}
            for data in image_data],
        "inter_cluster_weights": inter_cluster_weights
    }
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Clustering completed. Results saved to {output_json}")

if __name__ == "__main__":
    output_json = "clustered_memes.json"
    cluster_memes(input_json="embeddings_extraction_with_emotion.json", output_json="cluster_with_emotion.json")
    cluster_memes(input_json="embeddings_extraction_without_emotion.json", output_json="cluster_without_emotion.json")
