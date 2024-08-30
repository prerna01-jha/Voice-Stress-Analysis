# Voice-Stress-Analysis
Voice Stress Analysis (VSA), a technique that analyze speech patterns to detect psychological states like stress,  deceit, and emotions.
Explanation of what the code :

1. **Libraries and Dependencies:**
   - The code starts by importing various libraries such as `numpy`, `pandas`, `keras`, and others. These libraries are essential for data manipulation, building neural networks, and performing the training process.

2. **Loading and Preprocessing Data:**
   - Data is loaded using `pd.read_csv()` from a CSV file that likely contains audio features extracted from speech samples along with their corresponding labels (representing emotions).
   - The features and labels are separated into `X` (features) and `y` (labels) variables.
   - The labels `y` are then converted into categorical data (one-hot encoded) using `to_categorical()`. This is a common approach when you have multiple classes and you need to represent the labels in a format suitable for neural networks.

3. **Splitting the Data:**
   - The dataset is split into training and testing sets using `train_test_split()`. This allows the model to be trained on one portion of the data and then evaluated on a separate portion to assess its performance.

4. **Building the Model:**
   - A Sequential model from Keras is created. This type of model is a linear stack of layers, where each layer has one input and one output.
   - The layers added include:
     - **Dense layers:** Fully connected layers where each neuron in the layer is connected to every neuron in the previous layer.
     - **Dropout layers:** These help prevent overfitting by randomly setting a fraction of input units to 0 at each update during training time.
     - **Batch Normalization layers:** These standardize the inputs to a layer, improving training speed and stability.

5. **Compiling the Model:**
   - The model is compiled using the `adam` optimizer, `categorical_crossentropy` loss function (suitable for multi-class classification problems), and `accuracy` as the metric to monitor.

6. **Training the Model:**
   - The model is trained using the `fit()` method, which involves feeding the training data into the model and adjusting the weights to minimize the loss function.
   - The `validation_data` parameter is used to evaluate the model on the test set after each epoch (a complete pass through the training dataset).

7. **Evaluating the Model:**
   - The performance of the model is evaluated on the test set using the `evaluate()` method, which returns the loss and accuracy on the test data.

8. **Prediction:**
   - Finally, the model makes predictions on the test data using the `predict()` method, and the predicted classes are obtained by finding the index of the maximum value in the output vector for each input sample.

In essence, it is implementing a machine learning pipeline to train and evaluate a neural network model on a dataset, for emotion detection based on audio features. The goal is to classify each input sample into one of several emotion categories.
