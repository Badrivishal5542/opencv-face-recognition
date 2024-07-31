# Requirements
	- pip install PySimpleGUI
	- pip install numpy
	- pip install opencv_contrib_python

# Training bot
	To add new user 
	- python faces-save.py
	To recreate trainer.yml 
	- python faces-train.py

# Run Application
	- python faces.py


OpenCV Face Recognition 
Face recognition using OpenCV is a process that involves detecting faces in images or video frames, extracting unique features from these faces, and then comparing these features to a database of known faces to identify or verify the person's identity. Here’s a detailed explanation of the key concepts and processes involved:

1. Face Detection:
Objective: The first step in face recognition is to locate and detect all faces in the input image or video frame.
Methods: OpenCV provides several methods for face detection, including:
Haar Cascades: This is a machine learning-based approach where a cascade function is trained from a lot of positive and negative images. It’s efficient and can detect faces in real-time.
Deep Neural Networks (DNN): More advanced and accurate methods use pre-trained deep learning models like Single Shot Multibox Detector (SSD) or YOLO (You Only Look Once) to detect faces.
2. Feature Extraction:
Objective: Once faces are detected, the next step is to extract distinctive features from these faces. These features will be used to compare faces.
Methods:
Local Binary Patterns Histograms (LBPH): This method labels the pixels of an image by thresholding the neighborhood of each pixel and considers the result as a binary number. It's effective in varying lighting conditions and can recognize faces in real-time.
Eigenfaces: This method uses Principal Component Analysis (PCA) to reduce the dimensionality of the face images and extract the most important features.
Fisherfaces: Similar to Eigenfaces but uses Linear Discriminant Analysis (LDA) to find the linear combinations of features that best separate the classes (different faces).
3. Face Recognition:
Objective: The extracted features are compared to a database of known faces to recognize or verify the person’s identity.
Methods:
Distance Metrics: Techniques like Euclidean distance or Chi-square can be used to compare the similarity of feature vectors. The face with the smallest distance to the input face is considered the match.
Machine Learning Algorithms: More sophisticated methods might involve training machine learning models (e.g., SVMs) to classify faces based on their extracted features.
Process Flow
Detection Phase:

An image is fed into the system, where face detection algorithms locate all faces within the image.
Bounding boxes are created around detected faces.
Feature Extraction Phase:

For each detected face, feature extraction algorithms process the facial region to extract unique descriptors or features.
These features might include patterns, textures, or key points that uniquely represent the face.
Recognition/Classification Phase:

The extracted features are then compared against a pre-existing database of known face features.
The comparison yields a similarity score or confidence level indicating how closely the detected face matches a known face in the database.
Based on the similarity score, the system identifies or verifies the person’s identity.
Applications
Security and Surveillance: Used in security systems for identifying individuals in real-time.
Access Control: Used for biometric authentication to grant access to secure areas or devices.
Social Media: Used in platforms like Facebook for automatic tagging of users in photos.
Customer Interaction: Used in retail to personalize customer experiences by recognizing repeat customers.
Challenges
Variations in Lighting: Changes in lighting can significantly affect the appearance of the face, making it harder to recognize.
Pose Variation: The face might be at different angles or orientations, complicating recognition.
Occlusions: Items like glasses, hats, or masks can obscure parts of the face.
Aging: Over time, facial features can change due to aging.
OpenCV provides a robust framework for implementing face recognition systems, supporting various algorithms for detection, feature extraction, and recognition. By leveraging these tools, developers can create applications that can accurately and efficiently recognize faces in diverse conditions.






