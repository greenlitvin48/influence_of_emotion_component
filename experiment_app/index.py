import json
import random
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

with open("data/cluster_with_emotion.json", "r") as f:
    data_group3 = json.load(f)

with open("data/cluster_without_emotion.json", "r") as f:
    data_group8 = json.load(f)


@app.route("/")
def assign_group():
    """Randomly assign users to group 3 or 8 and redirect them"""
    group_id = random.choices([3, 8], weights=[50, 50], k=1)[0]
    return redirect(f"/{group_id}")


@app.route("/<int:group_id>")
def index(group_id):
    """Serve the main page for a specific group"""
    if group_id not in [3, 8]:
        return "Invalid Group", 404
    return render_template("index.html", group_id=group_id)


def get_dataset(group_id):
    """Retrieve the correct dataset based on group_id"""
    if group_id == 3:
        return data_group3["images"], data_group3["inter_cluster_weights"]
    elif group_id == 8:
        return data_group8["images"], data_group8["inter_cluster_weights"]
    else:
        return None, None


@app.route("/get_random_memes/<int:group_id>")
def get_random_memes(group_id):
    """Get 20 random memes from the dataset"""
    image_data, _ = get_dataset(group_id)
    if image_data is None:
        return jsonify({"error": "Invalid group"}), 400

    random_memes = random.sample(image_data, 20)
    return jsonify(random_memes)


@app.route("/find_similar/<int:group_id>", methods=["POST"])
def find_similar(group_id):
    """Find similar memes based on selected images and feedback scores (if provided)"""
    # preprocess images into dictionary (for faster lookups)
    image_data, inter_cluster_weights = get_dataset(group_id)
    image_dict = {img["image_name"]: img for img in image_data}

    if not image_data:
        return jsonify({"error": "Invalid group"}), 400

    # get all parameters
    selected_images = request.json["selected_images"]
    feedback_scores = request.json.get("feedback_scores", {})

    # collect features from selected images and feedback together
    all_features = []
    cluster_ids = []

    # 1. Process selected images
    for img_name in selected_images:
        if img_name in image_dict:
            img_data = image_dict[img_name]
            all_features.append(np.array(img_data["pca_features"]))
            cluster_ids.append(img_data["cluster_id"])

    # 2. Process feedback scores (NEW: Apply to recommended images
    for img_name, score in feedback_scores.items():
        if img_name in image_dict and score != 0:  # Only process non-zero feedback
            img_data = image_dict[img_name]
            # multiply by score directly to get weighted features
            weighted_features = np.array(img_data["pca_features"]) * score
            all_features.append(weighted_features)
            cluster_ids.append(img_data["cluster_id"])

    # calculate combined query features
    if not all_features:
        return jsonify({"error": "No valid input"}), 400

    query_features = np.mean(all_features, axis=0)
    query_cluster = max(
        set(cluster_ids), key=cluster_ids.count) if cluster_ids else None

    # find similar using cluster weight handling
    found_memes = search_memes(
        query_features=query_features,
        query_cluster=query_cluster,
        exclude_names=selected_images + [
            img_name
            for img_name, score in feedback_scores.items()
            if score < 0  # only exclude negative feedback memes
        ],
        image_data=image_data,
        inter_cluster_weights=inter_cluster_weights,
        top_k=5
    )

    return jsonify({
        "found_memes": format_results(found_memes)
    })


def search_memes(query_features, query_cluster, exclude_names, image_data, inter_cluster_weights, top_k=5):
    """Search memes using cluster weights"""
    similarities = []

    for data in image_data:
        if data["image_name"] in exclude_names:
            continue

        db_features = np.array(data["pca_features"])
        db_cluster = data["cluster_id"]

        # calculate base similarity
        embedding_sim = cosine_similarity(
            [query_features], [db_features])[0][0]

        # get cluster weight
        cluster_weight = inter_cluster_weights.get(
            query_cluster, {}).get(db_cluster, 0.0)

        # combine similarities
        combined_sim = embedding_sim * \
            (1 + cluster_weight)  # scale by cluster affinity

        similarities.append((combined_sim, data))

    # return top k results
    return sorted(similarities, key=lambda x: -x[0])[:top_k]


def format_results(results):
    """Helper to format results consistently"""
    return [{
        "image_name": d[1]["image_name"],
        "similarity": d[0],
        "base64": d[1]["image_base64"]
    } for d in results]


if __name__ == "__main__":
    app.run(debug=True)
