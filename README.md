README for CNN-Based Image Anomaly Detection
Overview

This Python script implements a Convolutional Neural Network (CNN) for anomaly detection in images. Specifically, it utilizes a U-Net architecture for segmenting normal, malignant, and benign data in a set of images.
Requirements

    Python 3.x
    TensorFlow
    scikit-learn
    OpenCV
    NumPy
    Matplotlib

Functionality

    Data Loading and Preprocessing: The script loads image data from specified directories, applies preprocessing like resizing and normalization, and creates masks for anomaly regions.

    Model Training: It employs a U-Net model, a type of CNN that's effective for image segmentation tasks. The model is trained on the preprocessed data.

    Evaluation and Testing: After training, the model's performance is evaluated using a test set. Accuracy and loss metrics are provided.

    Prediction and Visualization: The script predicts anomaly masks for new images and visualizes the results alongside the original images and true masks.

How to Use

    Set Data Paths: Modify the normal_data, malignant_data, and benign_data variables to point to your image directories.

    Load and Process Data: Call load_data() function to load and preprocess your images.

    Train the Model: Initialize and train the U-Net model using the processed data.

    Evaluate the Model: Evaluate the model's performance on the test set.

    Predict and Visualize: Predict masks for new images and visualize the results.

Important Notes

    Ensure all libraries are installed using pip install.
    Adjust the paths to the data folders according to your local setup.
    The script includes warnings related to the use of deprecated parameters (lr) in Keras optimizers. These should be replaced with learning_rate in future updates.

Sample Output

The script logs the training process, showing loss and accuracy for each epoch, and provides a final evaluation on the test dataset. It also displays visual comparisons of original images, true masks, and predicted masks for a sample of test images.
