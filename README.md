## About

Welcome to my AI Programming Showcase! This repository showcases my journey in implementing machine learning and deep learning algorithms, both from scratch and using libraries like scikit-learn, TensorFlow, Keras, and PyTorch. Each implementation includes Jupyter notebooks or Python scripts, applied to various datasets—generated or real-world—to demonstrate the algorithms. This is intended for learners and practitioners alike to explore AI concepts and implementations.

### Repository Structure

- **MachineLearning**: Traditional machine learning algorithms.
  - **K-Means**: Clustering.
  - **KNN**: K-Nearest Neighbors.
  - **NaiveBayes**: Naive Bayes classifiers.
  - **PrincipalComponentAnalysis**: PCA for dimensionality reduction.
  - **Regression**: Linear, Logistic, Polynomial, and Ridge Regression.
- **DeepLearning**: Deep learning implementations.
  - **Keras**: Keras-based solutions.
  - **Neural_Network_Implementation_using_NumPy**: Custom neural network with NumPy.
  - **Pytorch**: PyTorch implementations (CNNs, GANs).
  - **Tensorflow**: TensorFlow implementations (MLPs, CNNs, RNNs, GANs).

Each subdirectory contains code, notebooks, and occasionally datasets or model weights.

### Visualisation

| Problem                          | Description                                      | Implementation                                                                                                                    | Dataset                 |
|----------------------------------|--------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| Picking best computer game to try| K-Means clustering for visualizing top positions | [Python (raw)](AI-Implementations/MachineLearning/K-Means/raw_solution/K-Means_VideoGames_Raw.ipynb) | Kaggle - Video Game Sales |

## AI Programming Showcase

This section highlights my work with AI algorithms and frameworks, focusing on understanding their mechanics rather than solving complex problems. Implementations use classical or generated datasets.

### Raw Python

#### Machine Learning

| Algorithm           | Description                          | Implementation                                                                                                                    | Dataset               |
|---------------------|--------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|-----------------------|
| Linear Regression   | Basic regression                     | [Python (raw)](AI-Implementations/MachineLearning/Regression/LinearRegression/raw_solution/LinearRegression_Raw.ipynb) | Generated Numbers     |
| Ridge Regression    | Compared with Linear Regression      | [Python (raw)](AI-Implementations/MachineLearning/Regression/RidgeRegression/raw_solution/RidgeRegression_Raw.ipynb) | Generated Numbers     |
| Polynomial Regression | Degree 2 approximation             | [Python (raw)](AI-Implementations/MachineLearning/Regression/PolynomialRegression/raw_solution/PolynomialRegression_Degree2_Raw.ipynb) | Generated Numbers     |
| Polynomial Regression | Degree 3 approximation             | [Python (raw)](AI-Implementations/MachineLearning/Regression/PolynomialRegression/raw_solution/PolynomialRegression_Degree3_Raw.ipynb) | Generated Numbers     |
| KNN                 | Manhattan, Euclidean similarity      | [Python (raw)](AI-Implementations/MachineLearning/KNN/raw_solution/KNN_Iris_Raw.ipynb) | iris                  |
| PCA                 | Dimensionality reduction             | [Python (raw)](AI-Implementations/MachineLearning/PrincipalComponentAnalysis/PCA_Raw.ipynb) | Generated Numbers     |
| Naive Bayes         | Gaussian distribution                | [Python (raw)](AI-Implementations/MachineLearning/NaiveBayes/raw_solution/NaiveBayes_PimaIndiansDiabetes_raw.ipynb) | Pima Indian Diabetes  |
| Logistic Regression | Binary classification                | [Python (raw)](AI-Implementations/MachineLearning/Regression/LogisticRegression/raw_solution/LogisticRegression_Raw.ipynb) | Titanic Disaster Data |

#### Deep Learning

| Net Type | Problem            | Description         | Implementation                                                                                                                    | Dataset |
|----------|--------------------|---------------------|-----------------------------------------------------------------------------------------------------------------------------------|---------|
| MLP      | Digit Classification | 2-layers, mini-batch | [Python (raw)](AI-Implementations/DeepLearning/Raw/MLP_MNIST/MultilayerPerceptron-MNIST-Raw.ipynb) | MNIST   |

### scikit-learn

| Algorithm           | Description                          | Implementation                                                                                                                    | Dataset           |
|---------------------|--------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|-------------------|
| Linear Regression   | Basic regression                     | [Python (sklearn)](AI-Implementations/MachineLearning/Regression/LinearRegression/sklearn_solution/LinearRegression_Sklearn.ipynb) | Generated Numbers |
| Polynomial Regression | Degree 2 approximation             | [Python (sklearn)](AI-Implementations/MachineLearning/Regression/PolynomialRegression/sklearn_solution/PolynomialRegression_Degree2_Sklearn.ipynb) | Generated Numbers |
| Polynomial Regression | Degree 3 approximation             | [Python (sklearn)](AI-Implementations/MachineLearning/Regression/PolynomialRegression/sklearn_solution/PolynomialRegression_Degree3_Sklearn.ipynb) | Generated Numbers |
| KNN                 | Euclidean similarity                 | [Python (sklearn)](AI-Implementations/MachineLearning/KNN/sklearn_solution/KNN_Iris_Sklearn.ipynb) | iris              |

### TensorFlow

#### Machine Learning

| Algorithm         | Description            | Implementation                                                                                                                    | Dataset           |
|-------------------|------------------------|-----------------------------------------------------------------------------------------------------------------------------------|-------------------|
| Linear Regression | Basic regression       | [Python (Tensorflow)](AI-Implementations/MachineLearning/Regression/LinearRegression/tensorflow_solution/LinearRegression_Tensorflow.ipynb) | Generated Numbers |

#### Deep Learning

| Net Type | Problem                         | Description                            | Implementation                                                                                                                    | Dataset                   |
|----------|---------------------------------|----------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|---------------------------|
| MLP      | Digit Classification            | 2-layers, mini-batch, dropout          | [Python (Tensorflow)](AI-Implementations/DeepLearning/Tensorflow/MLP_MNIST/MultilayerPerceptron-MNIST-Tensorflow.ipynb) | MNIST                     |
| MLP      | Encrypting data with Autoencoder | 1-layer Encoder/Decoder, mini-batch    | [Python (Tensorflow)](AI-Implementations/DeepLearning/Tensorflow/Autoencoder_ImageEncriptionMNIST/MLP-Encryption-Autoencoder.ipynb) | MNIST                     |
| MLP      | Digit Classification            | Batch normalization, dropout           | [Python (Tensorflow)](AI-Implementations/DeepLearning/Tensorflow/CNN_MNIST/ConvNet-MNIST-Tensorflow-BN-tflayer.ipynb) | MNIST                     |
| CNN      | 10 Classes Color Images         | Dropout regularization                 | [Python (Tensorflow)](AI-Implementations/DeepLearning/Tensorflow/CNN_CIFAR10/ConvNet-CIFAR10-Tensorflow-tfnn.ipynb) | CIFAR-10                  |
| CNN      | 10 Classes Color Images         | Dropout regularization                 | [Python (Tensorflow)](AI-Implementations/DeepLearning/Tensorflow/CNN_CIFAR10/ConvNet-CIFAR10-Tensorflow-tflayer.ipynb) | CIFAR-10                  |
| CNN      | 10 Classes Color Images         | Batch normalization, dropout           | [Python (Tensorflow)](AI-Implementations/DeepLearning/Tensorflow/CNN_CIFAR10/ConvNet-CIFAR10-Tensorflow-BN-tflayer.ipynb) | CIFAR-10                  |
| RNN      | Simple Language Translator      | Basic RNN                              | [Python (Tensorflow)](AI-Implementations/DeepLearning/Tensorflow/RNN_FrenchEnglishTranslatior/dlnd_language_translation.ipynb) | French-English corpus |
| RNN      | "The Simpsons" Script Generation | Character-level RNN                    | [Python (Tensorflow)](AI-Implementations/DeepLearning/Tensorflow/RNN_SimpsonsScriptGenerator/dlnd_tv_script_generation.ipynb) | "The Simpsons" script     |
| DCGAN    | Generating Human Face Miniatures | Deep Convolutional GAN                 | [Python (Tensorflow)](AI-Implementations/DeepLearning/Tensorflow/GAN_CelebrityFaceGenerator/DC-GAN-FaceGeneration-Tensorflow.ipynb) | CelebA                    |

### Keras

| Net Type | Problem                         | Description                            | Implementation                                                                                                                    | Dataset           |
|----------|---------------------------------|----------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|-------------------|
| MLP      | Digit Classification            | 2-layers, mini-batch, BN               | [Python (Keras)](AI-Implementations/DeepLearning/Keras/MLP_MNIST/MLP%20MNIST%20-%20Keras%20Solution.ipynb) | MNIST             |
| MLP      | Clothes Images Classification   | 2-layers, mini-batch, BN               | [Python (Keras)](AI-Implementations/DeepLearning/Keras/MLP_FashionMnist/MLP%20Fashion%20MNIST%20-%20Keras%20Solution.ipynb) | Fashion MNIST     |
| MLP      | Letters Classification          | 2-layers, mini-batch, BN               | [Python (Keras)](AI-Implementations/DeepLearning/Keras/MLP_EMNIST/MLP%20EMNIST%20-%20Keras%20Solution.ipynb) | EMNIST            |
| MLP      | Review Sentiment Classification | Bag of Words                           | [Python (Keras)](AI-Implementations/DeepLearning/Keras/MLP_ImdbReviewSentimentPrediction/MLP%20IMDB%20Sentiment%20Analysis%20-%20Bag%20of%20Words%20-%20%20Keras%20Solution.ipynb) | IMDB Reviews      |
| MLP      | Boston House Prices Regression  | 1-layer, mini-batch                    | [Python (Keras)](AI-Implementations/DeepLearning/Keras/MLP_BostonHousePricesPrediction/MLP%20Boston%20House%20Prices%20-%20Keras%20Solution.ipynb) | Boston House Prices |
| CNN      | Ten Color Image Classes         | VGG15                                  | [Python (Keras)](AI-Implementations/DeepLearning/Keras/CNN_CIFAR10/CNN%20CIFAR10%20-%20Keras%20Solution.ipynb) | CIFAR10           |
| CNN      | Letter Classification           | 32x32x64x64, 512, BN                  | [Python (Keras)](AI-Implementations/DeepLearning/Keras/CNN_EMNIST/CNN%20EMNIST%20-%20Keras%20Solution.ipynb) | EMNIST            |
| CNN      | Clothes Images Classification   | 16x16x32x32, 256x128, BN              | [Python (Keras)](AI-Implementations/DeepLearning/Keras/CNN_FashionMNIST/CNN%20Fashion%20MNIST%20-%20Keras%20Solution.ipynb) | Fashion MNIST     |
| CNN      | Digit Classification            | 16x32x64, 128, BN                     | [Python (Keras)](AI-Implementations/DeepLearning/Keras/CNN_MNIST/CNN%20MNIST%20-%20Keras%20Solution.ipynb) | MNIST             |
| RNN      | Next Month Prediction           | LSTM(128)                             | [Python (Keras)](AI-Implementations/DeepLearning/Keras/RNN_MonthOrderPrediction/RNN%20Month%20Order%20-%20Keras%20Solution.ipynb) | Month Order       |
| RNN      | Shakespeare Sonnet's Generation | LSTM(700), LSTM(700)                  | [Python (Keras)](AI-Implementations/DeepLearning/Keras/RNN_ShakespeareGeneration/RNN%20Text%20Generation%20%5BShakespeare%5D%20-%20Keras%20Solution.ipynb) | Shakespeare's sonnets |

### PyTorch

#### CNN Architectures

| Architecture   | Description                          | Implementation                                                                                                                    |
|----------------|--------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| LeNet5         | Classic CNN                          | [Python (PyTorch)](AI-Implementations/DeepLearning/Pytorch/CNN_architectures/lenet5_pytorch.py) |
| InceptionNet   | Google’s Inception architecture      | [Python (PyTorch)](AI-Implementations/DeepLearning/Pytorch/CNN_architectures/pytorch_inceptionet.py) |
| VGG            | VGG network                          | [Python (PyTorch)](AI-Implementations/DeepLearning/Pytorch/CNN_architectures/pytorch_vgg_implementation.py) |
| EfficientNet   | Efficient CNN                        | [Python (PyTorch)](AI-Implementations/DeepLearning/Pytorch/CNN_architectures/pytorch_efficientnet.py) |
| ResNet         | Residual Network                     | [Python (PyTorch)](AI-Implementations/DeepLearning/Pytorch/CNN_architectures/pytorch_resnet.py) |

#### GANs

| GAN Type   | Description                          | Implementation                                                                                                                    |
|------------|--------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| SimpleGAN  | Fully connected GAN                  | [Python (PyTorch)](AI-Implementations/DeepLearning/Pytorch/GANs/1.%20SimpleGAN/fc_gan.py) |
| DCGAN      | Deep Convolutional GAN               | [Python (PyTorch)](AI-Implementations/DeepLearning/Pytorch/GANs/2.%20DCGAN/model.py), [train.py](AI-Implementations/DeepLearning/Pytorch/GANs/2.%20DCGAN/train.py) |
| WGAN       | Wasserstein GAN                      | [Python (PyTorch)](AI-Implementations/DeepLearning/Pytorch/GANs/3.%20WGAN/model.py), [train.py](AI-Implementations/DeepLearning/Pytorch/GANs/3.%20WGAN/train.py) |
| WGAN-GP    | WGAN with Gradient Penalty           | [Python (PyTorch)](AI-Implementations/DeepLearning/Pytorch/GANs/4.%20WGAN-GP/model.py), [train.py](AI-Implementations/DeepLearning/Pytorch/GANs/4.%20WGAN-GP/train.py), [utils.py](AI-Implementations/DeepLearning/Pytorch/GANs/4.%20WGAN-GP/utils.py) |
| CycleGAN   | Image-to-image translation           | [Python (PyTorch)](AI-Implementations/DeepLearning/Pytorch/GANs/5.CycleGAN/config.py), [dataset.py](AI-Implementations/DeepLearning/Pytorch/GANs/5.CycleGAN/dataset.py), [generator_model.py](AI-Implementations/DeepLearning/Pytorch/GANs/5.CycleGAN/generator_model.py), [discriminator_model.py](AI-Implementations/DeepLearning/Pytorch/GANs/5.CycleGAN/discriminator_model.py), [train.py](AI-Implementations/DeepLearning/Pytorch/GANs/5.CycleGAN/train.py), [utils.py](AI-Implementations/DeepLearning/Pytorch/GANs/5.CycleGAN/utils.py) |
| Pix2Pix    | Paired image-to-image translation    | [Python (PyTorch)](AI-Implementations/DeepLearning/Pytorch/GANs/6.Pix2Pix/config.py), [dataset.py](AI-Implementations/DeepLearning/Pytorch/GANs/6.Pix2Pix/dataset.py), [generator_model.py](AI-Implementations/DeepLearning/Pytorch/GANs/6.Pix2Pix/generator_model.py), [discriminator_model.py](AI-Implementations/DeepLearning/Pytorch/GANs/6.Pix2Pix/discriminator_model.py), [train.py](AI-Implementations/DeepLearning/Pytorch/GANs/6.Pix2Pix/train.py), [utils.py](AI-Implementations/DeepLearning/Pytorch/GANs/6.Pix2Pix/utils.py) |

### Neural Network Implementation using NumPy

A custom neural network built from scratch using NumPy to explore deep learning fundamentals. It includes:

- **Training Scripts**: 
  - [train_BGD.py](AI-Implementations/DeepLearning/Neural_Network_Implementation_using_NumPy/train_BGD.py) (Batch Gradient Descent)
  - [train_SGD.py](AI-Implementations/DeepLearning/Neural_Network_Implementation_using_NumPy/train_SGD.py) (Stochastic Gradient Descent)
  - [train.py](AI-Implementations/DeepLearning/Neural_Network_Implementation_using_NumPy/train.py) (General training)
- **Testing Script**: [test.py](AI-Implementations/DeepLearning/Neural_Network_Implementation_using_NumPy/test.py)
- **Utilities**: Data loading, loss functions (CrossEntropy, MSE), model definitions, and plotting.
- **Visualizations**: Accuracy/loss plots and confusion matrices for train, validation, and test data.

This serves as an educational resource for understanding neural networks without high-level frameworks.

