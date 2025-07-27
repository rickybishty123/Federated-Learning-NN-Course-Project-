<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Federated Learning Project README</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f4f7f6;
            color: #333;
        }
        .container {
            max-width: 960px;
            margin: 0 auto;
            padding: 2rem;
            background-color: #ffffff;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-radius: 0.75rem; /* rounded-xl */
        }
        h1, h2, h3, h4, h5, h6 {
            font-weight: bold;
            color: #2c3e50;
        }
        h1 { font-size: 2.5rem; margin-bottom: 1.5rem; }
        h2 { font-size: 2rem; margin-top: 2rem; margin-bottom: 1rem; border-bottom: 2px solid #eee; padding-bottom: 0.5rem; }
        h3 { font-size: 1.5rem; margin-top: 1.5rem; margin-bottom: 0.75rem; }
        ul {
            list-style-type: disc;
            margin-left: 1.5rem;
            margin-bottom: 1rem;
        }
        ol {
            list-style-type: decimal;
            margin-left: 1.5rem;
            margin-bottom: 1rem;
        }
        pre {
            background-color: #ecf0f1;
            padding: 1rem;
            border-radius: 0.5rem;
            overflow-x: auto;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
            font-size: 0.875rem;
            line-height: 1.4;
            margin-bottom: 1rem;
        }
        code {
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
            background-color: #e0e6e8;
            padding: 0.2em 0.4em;
            border-radius: 0.3em;
            font-size: 0.875rem;
        }
        a {
            color: #3498db;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        p {
            margin-bottom: 1rem;
            line-height: 1.6;
        }
    </style>
</head>
<body class="p-4 sm:p-6 md:p-8 lg:p-10">
    <div class="container">
        <h1 class="text-center text-gray-800">Federated Learning with RNN for Next Word Prediction</h1>

        <p class="text-lg text-center text-gray-600 mb-8">
            This repository contains a Python implementation of a federated learning system using a Recurrent Neural Network (RNN) with LSTM layers for next word prediction. The project simulates a distributed learning environment where multiple clients collaboratively train a global model without sharing their raw data.
        </p>

        <h2 id="table-of-contents">Table of Contents</h2>
        <ul class="list-disc ml-6 mb-4">
            <li><a href="#introduction">Introduction</a></li>
            <li><a href="#features">Features</a></li>
            <li><a href="#project-structure">Project Structure</a></li>
            <li><a href="#how-it-works">How it Works</a></li>
            <li><a href="#setup-and-installation">Setup and Installation</a></li>
            <li><a href="#usage">Usage</a></li>
            <li><a href="#results">Results</a></li>
            <li><a href="#contributing">Contributing</a></li>
            <li><a href="#license">License</a></li>
            <li><a href="#acknowledgements">Acknowledgements</a></li>
        </ul>

        <h2 id="introduction">Introduction</h2>
        <p>
            Federated Learning is a decentralized machine learning approach that enables multiple clients to collaboratively train a shared global model while keeping their training data local. This project demonstrates federated learning for a natural language processing (NLP) task: next word prediction. An RNN model with LSTM layers is used to learn patterns in text data distributed across various simulated clients.
        </p>

        <h2 id="features">Features</h2>
        <ul class="list-disc ml-6 mb-4">
            <li><strong>Federated Learning Simulation:</strong> Simulates a federated learning setup with a central server and multiple clients.</li>
            <li><strong>RNN with LSTM:</strong> Utilizes a Keras RNN model with LSTM layers for sequence modeling.</li>
            <li><strong>Next Word Prediction:</strong> Trains the model to predict the next word in a given sequence.</li>
            <li><strong>Weighted Averaging Aggregation:</strong> The server aggregates client model weights using weighted averaging based on client sample sizes.</li>
            <li><strong>Configurable Parameters:</strong> Easily adjust the number of clients, training rounds, and client participation fraction.</li>
            <li><strong>Separate Test Dataset:</strong> Evaluates the global model on a distinct test dataset to assess generalization performance.</li>
        </ul>

        <h2 id="project-structure">Project Structure</h2>
        <p>The project consists of a single Python script:</p>
        <ul class="list-disc ml-6 mb-4">
            <li><code>federated_learning_project.py</code>: Contains the core implementation of the RNN model, Federated Server, Federated Client, data loading, and the main federated learning loop.</li>
        </ul>

        <h2 id="how-it-works">How it Works</h2>
        <ol class="list-decimal ml-6 mb-4">
            <li><strong>Data Loading:</strong> Sample text data is loaded and split among a specified number of simulated clients. A separate test dataset is also loaded for global model evaluation.</li>
            <li><strong>Tokenizer Initialization:</strong> A <code>Tokenizer</code> from Keras is used to convert text into sequences of integers, creating a vocabulary.</li>
            <li><strong>Model Definition (<code>RNNModel</code>):</strong>
                <ul class="list-disc ml-6 mt-2">
                    <li>An <code>RNNModel</code> class is defined using <code>tf.keras.Model</code>.</li>
                    <li>It includes an <code>Embedding</code> layer to convert word indices into dense vectors, an <code>LSTM</code> layer to capture sequential dependencies, and a <code>Dense</code> layer with <code>softmax</code> activation for predicting the next word's probability distribution.</li>
                </ul>
            </li>
            <li><strong>Federated Server (<code>FederatedServer</code>):</strong>
                <ul class="list-disc ml-6 mt-2">
                    <li>Initializes a <code>global_model</code> (an instance of <code>RNNModel</code>).</li>
                    <li><code>aggregate_weights()</code>: Receives updated weights from selected clients and performs a weighted average to update the <code>global_model</code>. The weights are averaged based on the number of samples each client trained on.</li>
                    <li><code>send_model()</code>: Sends the current <code>global_model</code>'s weights to clients.</li>
                    <li><code>evaluate()</code>: Evaluates the <code>global_model</code> on a separate test dataset.</li>
                </ul>
            </li>
            <li><strong>Federated Client (<code>FederatedClient</code>):</strong>
                <ul class="list-disc ml-6 mt-2">
                    <li>Each client holds its own local data.</li>
                    <li><code>preprocess_data()</code>: Converts the client's raw text data into input-output sequences suitable for training the RNN.</li>
                    <li><code>client_update()</code>:
                        <ul class="list-disc ml-6 mt-2">
                            <li>Receives the <code>global_weights</code> from the server.</li>
                            <li>Sets its local model's weights to the <code>global_weights</code>.</li>
                            <li>Trains its local model on its private data for a specified number of <code>local_epochs</code>.</li>
                            <li>Returns the updated local model weights to the server.</li>
                        </ul>
                    </li>
                </ul>
            </li>
            <li><strong>Federated Learning Loop:</strong>
                <ul class="list-disc ml-6 mt-2">
                    <li>The <code>federated_learning()</code> function orchestrates the entire process.</li>
                    <li>For a defined number of <code>num_rounds</code>:
                        <ul class="list-disc ml-6 mt-2">
                            <li>A <code>fraction_clients</code> of total clients are randomly selected for participation in the current round.</li>
                            <li>Each selected client performs <code>client_update()</code>: downloads the global model, trains it locally on its data, and uploads its updated weights.</li>
                            <li>The server performs <code>aggregate_weights()</code>: combines the received client weights to update the global model.</li>
                            <li>The global model is evaluated on the separate test dataset to track performance.</li>
                        </ul>
                    </li>
                </ul>
            </li>
            <li><strong>Next Word Prediction:</strong> After all training rounds, the final global model is used to predict the next word for a given input phrase.</li>
        </ol>

        <h2 id="setup-and-installation">Setup and Installation</h2>
        <p>To run this project, you need Python and TensorFlow installed.</p>
        <ol class="list-decimal ml-6 mb-4">
            <li><strong>Clone the repository (if applicable):</strong>
                <pre><code>git clone https://github.com/your-username/federated-learning-nlp.git
cd federated-learning-nlp
</code></pre>
            </li>
            <li><strong>Install dependencies:</strong>
                <pre><code>pip install numpy tensorflow
</code></pre>
            </li>
        </ol>

        <h2 id="usage">Usage</h2>
        <p>To run the federated learning simulation, simply execute the Python script:</p>
        <pre><code>python federated_learning_project.py
</code></pre>
        <p>You can modify the parameters in the <code>federated_learning()</code> function call at the end of the script to experiment with different configurations:</p>
        <pre><code>federated_learning(num_clients=2, num_rounds=50, fraction_clients=0.25)
</code></pre>
        <ul class="list-disc ml-6 mb-4">
            <li><code>num_clients</code>: The total number of simulated clients.</li>
            <li><code>num_rounds</code>: The total number of communication rounds between clients and the server.</li>
            <li><code>fraction_clients</code>: The fraction of clients to be selected in each round for training.</li>
            <li><code>sequence_length</code>: The length of input sequences for the RNN model.</li>
        </ul>
        <p>The script will print the test accuracy after each round and the predicted next word after the training is complete.</p>

        <h2 id="results">Results</h2>
        <p>During the execution, you will see output similar to this, showing the progress of the federated learning:</p>
        <pre><code>Round 1
Test Accuracy after round 1: 0.0000
Round 2
Test Accuracy after round 2: 0.0000
...
Round 50
Test Accuracy after round 50: 0.8000
The predicted word after "I am going" is: to
</code></pre>
        <p>The accuracy will generally improve over rounds as the global model learns from the distributed data. The final prediction demonstrates the model's ability to complete a sentence.</p>

        <h2 id="contributing">Contributing</h2>
        <p>Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.</p>

        <h2 id="license">License</h2>
        <p>This project is open-sourced under the MIT License. See the <code>LICENSE</code> file for more details.</p>

        <h2 id="acknowledgements">Acknowledgements</h2>
        <ul class="list-disc ml-6 mb-4">
            <li>TensorFlow and Keras for providing the deep learning framework.</li>
            <li>The concept of Federated Learning, pioneered by Google.</li>
        </ul>
    </div>
</body>
</html>
