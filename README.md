# Visual Product Matcher


  The Visual Product Matcher Project is designed to identify and retrieve images from a dataset that are visually similar to a given query image. This is widely used in e-commerce, content-based image retrieval, and multimedia search engines.  
The main idea is to represent images as feature vectors in a high-dimensional space. Similar photos are those whose feature vectors are "close" to each other based on a chosen similarity metric.  
  
  
  For the dataset, the images are downloaded from the Bing search engine using the open-source Python libraries. The dataset is organized in a folder structure with images stored in a single folder. The file format of the dataset is JPG format, and each image is preprocessed into a **224×224** fixed image size to match the input size of the feature extraction model.  
  Each image in the dataset is passed through a pre-trained **MobileNetV2** to generate a feature vector. MobileNetV2 is the base model, which is a very lightweight and efficient algorithm for the similarity product matcher.  The similarity is fetch using the Cosine similarity measurement. It measures the angle between vectors and is ideal for normalized features.  Once similarity is computed, the system ranks the images in the dataset according to their similarity scores and returns the top closest matches.
## Core Concept
  - Feature Extraction
  - Similarity Measurement
  - Ranking and Retrieval
## Workflow 
1. Dataset Preparation: Collect, organize, and preprocess images.
2. Feature Extraction: Convert all images into feature vectors.
3. Query Processing: Extract the feature vector from the uploaded image.
4. Similarity Search: Compute similarity and rank dataset images.
5. Result Display: Show top similar images in a grid layout.
## Website Link





