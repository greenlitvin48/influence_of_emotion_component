# Emotion-aware Multimodal Meme Retrieval

This public repository is an application to bachelor thesis "Emotion-aware Multimodal Meme Retrieval" written by Aleksandr Litvin.
The pdf file with the thesis can be found in this (root) folder.

## Table of Contents
- File structure
- Usage
- Contributing
- License
- Authors & Acknowledgments
- Contact Information

## File structure
### data
Contains data including zip-archive of jsons with clustered data for both approaches.
### experiment_app
Contains the Flask application which was used for user experiment (user interface and retrieval method).
### features_extraction_and_clustering
Contains the code of the preprocessing part: text extraction (the most time consuming), features extraction and PCA, clustering.

## Usage
### To run the preprocessing
1. Firstly feature_extraction_and_PCA.py (adjust file names and folder root) to extract text and filter your dataset
2. Then peform feature_extraction_and_PCA.py
3. After than clustering_function.py to cluster the retrieved PCA embeddings

### To run experiment application
The entry point for the Flask application is the index.py file. Depending on your system, you may need to create a virtual environment. Command on Linux may vary, to run in Windows cmd:

`set FLASK_APP=index.py`

`flask run`


## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Authors & Acknowledgments

The thesis is written by Aleksandr Litvin, and supervised by Christopher Bagdon and Prof. Dr. Roman Klinger.

## Contact Information

For all questions please contact aleksandr.litvin@stud.uni-bamberg.de (university email) or greenlitvin48@gmail.com (private email)
