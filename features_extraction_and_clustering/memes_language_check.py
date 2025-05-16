import os
import json
import easyocr
import shutil

def extract_text_from_images(image_folder, output_json):
    reader = easyocr.Reader(['en'])  # initialize EasyOCR reader
    results = []

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff')):
            image_path = os.path.join(image_folder, filename)
            text = reader.readtext(image_path, detail=0)  # Extract text without details
            results.append({"image": filename, "text": " ".join(text)})

    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)


def move_images_by_keywords(json_file, source_folder, target_folder, keywords):
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    os.makedirs(target_folder, exist_ok=True)

    for item in data:
        if any(word.lower() in item["text"].lower() for keyword in keywords for word in
               [keyword, keyword + 's', keyword + 'ed', keyword + 'ing', keyword + 'er']):
            source_path = os.path.join(source_folder, item["image"])
            target_path = os.path.join(target_folder, item["image"])
            if os.path.exists(source_path):
                shutil.move(source_path, target_path)
                print(f"Moved: {item['image']}")


if __name__ == "__main__":
    image_folder = "path/to/your/memes_dataset"
    output_json = "extracted_text_for_check.json"

    # slurs, vulgar, racist or stigmatizing words
    questionable_keywords = ["fuck", "bitch", "bitches", "black", "fag", "gay", "shoot", "black", "asian",
                             "nigger",
                             "chink",
                             "wetback",
                             "gook",
                             "spic",
                             "raghead",
                             "redskin",
                             "bitch",
                             "slut",
                             "cunt",
                             "dumb blonde",
                             "feminazi",
                             "he-man",
                             "rape",
                             "she asked for it",
                             "murder",
                             "molester",
                             "assault",
                             "fag",
                             "tranny",
                             "retard",
                             "psycho",
                             "schizo",
                             "bum"
                             ]
    nationalities = [
        "American",
        "Argentinian",
        "Australian",
        "Belgian",
        "Brazilian",
        "Bulgarian",
        "Canadian",
        "Chilean",
        "Chinese",
        "Colombian",
        "Croatian",
        "Cuban",
        "Danish",
        "Dutch",
        "Egyptian",
        "English",
        "Estonian",
        "Finnish",
        "French",
        "German",
        "Greek",
        "Hungarian",
        "Indian",
        "Indonesian",
        "Iranian",
        "Iraqi",
        "Irish",
        "Italian",
        "Japanese",
        "Korean",
        "Mexican",
        "Moroccan",
        "Nepalese",
        "New Zealander",
        "Nigerian",
        "Norwegian",
        "Pakistani",
        "Peruvian",
        "Philippine",
        "Polish",
        "Portuguese",
        "Romanian",
        "Russian",
        "Saudi",
        "Scottish",
        "Singaporean",
        "South African",
        "Spanish",
        "Swedish",
        "Swiss",
        "Taiwanese",
        "Thai",
        "Turkish",
        "Ukrainian",
        "Venezuelan",
        "Vietnamese"
    ]
    # if not acceptable words are found in the meme's text, move the meme into subfolder not_acceptable_memes
    move_images_by_keywords(output_json, image_folder, image_folder+'/not_acceptable_memes', questionable_keywords)
    print("Matching images with inappropriate words moved successfully.")
    move_images_by_keywords(output_json, image_folder, image_folder+'/not_acceptable_memes', nationalities)
    print("Matching images with nationalities in there moved successfully.")
