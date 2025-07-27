import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Define RNNModel with LSTM
class RNNModel(tf.keras.Model):
    def init(self, vocab_size, sequence_length):
        super(RNNModel, self).init()
        self.embedding = Embedding(input_dim=vocab_size, output_dim=64, input_length=sequence_length)
        self.lstm = tf.keras.layers.LSTM(64, return_sequences=False)
        self.dense = Dense(vocab_size, activation='softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm(x)
        return self.dense(x)


# Simulate a Federated Server
class FederatedServer:
    def init(self, vocab_size, sequence_length):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        
        # Create and initialize the global model
        self.global_model = RNNModel(vocab_size, sequence_length)
        # Dummy input to initialize the model's weights
        dummy_input = np.zeros((1, sequence_length))  # batch_size=1, sequence_length
        self.global_model(dummy_input)  # Call the model to build it
        self.global_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def aggregate_weights(self, client_weights_list, client_sample_sizes):
        # Weighted averaging of client weights based on their sample sizes
        total_samples = sum(client_sample_sizes)
        new_weights = [np.zeros_like(weight) for weight in client_weights_list[0]]

        for i, client_weights in enumerate(client_weights_list):
            for j, weight in enumerate(client_weights):
                new_weights[j] += (client_sample_sizes[i] / total_samples) * weight

        # Set the aggregated weights back to the global model
        self.global_model.set_weights(new_weights)

    def send_model(self):
        return self.global_model.get_weights()

    def evaluate(self, tokenizer, test_data, sequence_length):
        input_sequences, output_words = test_data.preprocess_data(tokenizer)
        loss, accuracy = self.global_model.evaluate(input_sequences, output_words, verbose=0)
        return loss, accuracy

# Simulate a Federated Client
class FederatedClient:
    def init(self, data, vocab_size, sequence_length, batch_size=32, local_epochs=3, learning_rate=0.001):
        self.data = data
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        
        self.model = RNNModel(vocab_size, sequence_length)
        # Dummy input to initialize the model's weights
        dummy_input = np.zeros((1, sequence_length))  # batch_size=1, sequence_length
        self.model(dummy_input)  # Call the model to build it
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def preprocess_data(self, tokenizer):
        # Convert text to sequences of integers
        sequence_data = tokenizer.texts_to_sequences([self.data])[0]
        
        # Prepare input-output sequences
        input_sequences = []
        output_words = []
        for i in range(self.sequence_length, len(sequence_data)):
            input_sequences.append(sequence_data[i-self.sequence_length:i])
            output_words.append(sequence_data[i])
        
        return np.array(input_sequences), np.array(output_words)

    def client_update(self, global_weights, tokenizer):
        # Set the model's weights to the current global weights
        self.model.set_weights(global_weights)
        # Preprocess data
        input_sequences, output_words = self.preprocess_data(tokenizer)
        # Train locally for specified number of epochs
        self.model.fit(input_sequences, output_words, epochs=self.local_epochs, batch_size=self.batch_size, verbose=0)
        # Return the updated weights after local training
        return self.model.get_weights()

# Function to load data
def load_data():
    # Expanded sample text data
    data = [
        "I am going to the store to buy some groceries.",
        "I am going to the park to relax.",
        "I am going to the office for work.",
        "I am walking to the bus stop.",
        "I am visiting the museum.",
        "She is reading a book in the library.",
        "He is running in the park every morning.",
        "They are planning a vacation to the mountains.",
        "We are preparing dinner for our friends.",
        "The cat is sleeping on the couch.",
        "He is fixing his car in the garage.",
        "She is attending a yoga class in the evening.",
        "The children are playing soccer in the yard.",
        "I am learning to play the guitar.",
        "We are watching a movie tonight.",
        "The teacher is explaining the math problem to the students.",
        "He is painting the walls of his new house.",
        "She is baking a cake for the party.",
        "I am traveling to Japan next month.",
        "They are building a new shopping mall downtown.",
        "He is writing a novel about his adventures.",
        "She is gardening in her backyard every weekend.",
        "The dog is barking at the strangers outside.",
        "We are hiking in the forest this weekend.",
        "She is sewing a new dress for the event.",
        "He is swimming in the pool every morning.",
        "The team is preparing for the upcoming tournament.",
        "They are launching a new app next year.",
        "She is learning French for her trip to Paris.",
        "I am renovating my kitchen next month.",
        "He is practicing his speech for the conference.",
        "We are organizing a charity event next weekend.",
        "The chef is preparing a special meal for the guests.",
        "She is adopting a puppy from the shelter.",
        "He is designing a new logo for the company.",
        "They are discussing the project timeline in the meeting.",
        "We are booking flights for our summer vacation.",
        "She is hosting a dinner party for her colleagues.",
        "The musician is composing a new song for the album.",
        "He is training for a marathon next year.",
        "They are decorating their house for the holidays.",
        "She is sketching a portrait of her friend.",
        "He is fixing the roof of his house.",
        "The scientist is researching a cure for the disease.",
        "She is planting flowers in her garden.",
        "They are rehearsing for their school play.",
        "I am reading a book about artificial intelligence.",
        "He is building a treehouse in the backyard."
    ]
    return data

# Separate function for test data
def load_test_data():
    return [
        "The dog is running in the park.",
        "She is writing an article for the magazine.",
        "They are rehearsing for their school play.",
        "I am reading a book about artificial intelligence.",
        "He is building a treehouse in the backyard."
        
    ]

def predict_next_word(model, tokenizer, input_text, sequence_length):
    # Tokenize the input text
    input_sequence = tokenizer.texts_to_sequences([input_text])[0]

    # If the input sequence is longer than the expected sequence length, truncate it
    if len(input_sequence) > sequence_length:
        input_sequence = input_sequence[-sequence_length:]

    # If the input sequence is shorter, pad it to match the sequence length
    input_sequence = np.pad(input_sequence, (sequence_length - len(input_sequence), 0), 'constant')

    # Reshape input to be compatible with the model
    input_sequence = np.array(input_sequence).reshape(1, sequence_length)

    # Predict the next word's token
    predicted_probs = model.predict(input_sequence, verbose=0)
    predicted_word_index = np.argmax(predicted_probs)

    # Find the word that corresponds to the predicted index
    predicted_word = tokenizer.index_word.get(predicted_word_index, "<unknown>")

    return predicted_word

# Main Federated Learning Loop for RNN
def federated_learning(num_clients=3, num_rounds=5, fraction_clients=0.6, sequence_length=3):
    # Load and split data among clients
    data = load_data()
    test_data = load_test_data()  # Load the separate test dataset

    # Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data + test_data)  # Include both training and test data for a consistent tokenizer

    vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size

    # Split data among clients
    client_data_splits = np.array_split(data, num_clients)

    # Initialize clients
    clients = [FederatedClient(client_data_splits[i][0], vocab_size, sequence_length) for i in range(num_clients)]

    # Initialize server
    server = FederatedServer(vocab_size, sequence_length)

    # Preprocess test data
    test_client = FederatedClient(test_data[0], vocab_size, sequence_length)

    # Training rounds
    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}")

        # Select a random subset of clients (fraction of total clients)
        num_selected_clients = max(1, int(fraction_clients * num_clients))
        selected_clients = random.sample(clients, num_selected_clients)

        client_weights = []
        client_sample_sizes = []

        # Clients train locally and send their weights to the server
        for client in selected_clients:
            client_sample_size = len(client.data.split())
            client_sample_sizes.append(client_sample_size)

            # Get the global model weights, perform local training, and send the new weights
            updated_weights = client.client_update(server.send_model(), tokenizer)
            client_weights.append(updated_weights)

        # Server aggregates the client weights
        server.aggregate_weights(client_weights, client_sample_sizes)

        # Evaluate the global model on the separate test dataset
        loss, accuracy = server.evaluate(tokenizer, test_client, sequence_length)
        print(f"Test Accuracy after round {round_num + 1}: {accuracy:.4f}")

    # After training rounds, use the global model to predict the next word after "I am"
    input_text = "I am going"
    predicted_word = predict_next_word(server.global_model, tokenizer, input_text, sequence_length)
    print(f'The predicted word after "{input_text}" is: {predicted_word}')

# Run the federated learning
federated_learning(num_clients=2, num_rounds=50, fraction_clients=0.25)