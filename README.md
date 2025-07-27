Federated Learning with RNN for Next Word Prediction

This repository contains a Python implementation of a federated learning system using a Recurrent Neural Network (RNN) with LSTM layers for next word prediction. The project simulates a distributed learning environment where multiple clients collaboratively train a global model without sharing their raw data.
Table of Contents

    Introduction

    Features

    Project Structure

    How it Works

    Setup and Installation

    Usage

    Results

    Contributing

    License

    Acknowledgements

Introduction

Federated Learning is a decentralized machine learning approach that enables multiple clients to collaboratively train a shared global model while keeping their training data local. This project demonstrates federated learning for a natural language processing (NLP) task: next word prediction. An RNN model with LSTM layers is used to learn patterns in text data distributed across various simulated clients.
Features

    Federated Learning Simulation: Simulates a federated learning setup with a central server and multiple clients.

    RNN with LSTM: Utilizes a Keras RNN model with LSTM layers for sequence modeling.

    Next Word Prediction: Trains the model to predict the next word in a given sequence.

    Weighted Averaging Aggregation: The server aggregates client model weights using weighted averaging based on client sample sizes.

    Configurable Parameters: Easily adjust the number of clients, training rounds, and client participation fraction.

    Separate Test Dataset: Evaluates the global model on a distinct test dataset to assess generalization performance.

Project Structure

The project consists of a single Python script:

    federated_learning_project.py: Contains the core implementation of the RNN model, Federated Server, Federated Client, data loading, and the main federated learning loop.

How it Works

    Data Loading: Sample text data is loaded and split among a specified number of simulated clients. A separate test dataset is also loaded for global model evaluation.

    Tokenizer Initialization: A Tokenizer from Keras is used to convert text into sequences of integers, creating a vocabulary.

    Model Definition (RNNModel):

        An RNNModel class is defined using tf.keras.Model.

        It includes an Embedding layer to convert word indices into dense vectors, an LSTM layer to capture sequential dependencies, and a Dense layer with softmax activation for predicting the next word's probability distribution.

    Federated Server (FederatedServer):

        Initializes a global_model (an instance of RNNModel).

        aggregate_weights(): Receives updated weights from selected clients and performs a weighted average to update the global_model. The weights are averaged based on the number of samples each client trained on.

        send_model(): Sends the current global_model's weights to clients.

        evaluate(): Evaluates the global_model on a separate test dataset.

    Federated Client (FederatedClient):

        Each client holds its own local data.

        preprocess_data(): Converts the client's raw text data into input-output sequences suitable for training the RNN.

        client_update():

            Receives the global_weights from the server.

            Sets its local model's weights to the global_weights.

            Trains its local model on its private data for a specified number of local_epochs.

            Returns the updated local model weights to the server.

    Federated Learning Loop:

        The federated_learning() function orchestrates the entire process.

        For a defined number of num_rounds:

            A fraction_clients of total clients are randomly selected for participation in the current round.

            Each selected client performs client_update(): downloads the global model, trains it locally on its data, and uploads its updated weights.

            The server performs aggregate_weights(): combines the received client weights to update the global model.

            The global model is evaluated on the separate test dataset to track performance.

    Next Word Prediction: After all training rounds, the final global model is used to predict the next word for a given input phrase.

Setup and Installation

To run this project, you need Python and TensorFlow installed.

    Clone the repository (if applicable):

    git clone https://github.com/your-username/federated-learning-nlp.git
    cd federated-learning-nlp

    Install dependencies:

    pip install numpy tensorflow

Usage

To run the federated learning simulation, simply execute the Python script:

python federated_learning_project.py

You can modify the parameters in the federated_learning() function call at the end of the script to experiment with different configurations:

federated_learning(num_clients=2, num_rounds=50, fraction_clients=0.25)

    num_clients: The total number of simulated clients.

    num_rounds: The total number of communication rounds between clients and the server.

    fraction_clients: The fraction of clients to be selected in each round for training.

    sequence_length: The length of input sequences for the RNN model.

The script will print the test accuracy after each round and the predicted next word after the training is complete.
Results

During the execution, you will see output similar to this, showing the progress of the federated learning:

Round 1
Test Accuracy after round 1: 0.0000
Round 2
Test Accuracy after round 2: 0.0000
...
Round 50
Test Accuracy after round 50: 0.8000
The predicted word after "I am going" is: to

The accuracy will generally improve over rounds as the global model learns from the distributed data. The final prediction demonstrates the model's ability to complete a sentence.
Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.
License

This project is open-sourced under the MIT License. See the LICENSE file for more details.
Acknowledgements

    TensorFlow and Keras for providing the deep learning framework.

    The concept of Federated Learning, pioneered by Google.
