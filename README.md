# Sentiment Analysis with TensorFlow

This program implements a sentiment analysis model using TensorFlow 2.x to classify text comments as positive or negative.

## Requirements

- TensorFlow 2.x
- pandas
- numpy
- matplotlib

## Dataset

The program uses a CSV file named `sentiment_data.csv` containing:
- Comment column: Text comments
- Sentiment column: Binary sentiment labels

## Model Architecture

The neural network consists of:
1. Embedding layer (vocab_size=10000, embedding_dim=16)
2. Global Average Pooling 1D layer
3. Dense layer with 24 units and ReLU activation
4. Output Dense layer with sigmoid activation

## Key Parameters

```python
vocab_size = 10000
embedding_dim = 16
max_length = 100
training_size = 20000
num_epochs = 30
```

## Usage

1. Upload the sentiment_data.csv file
2. The program will:
   - Preprocess the text data using tokenization
   - Split data into training and testing sets
   - Train the model
   - Generate visualization plots for accuracy and loss
   - Create embeddings files (vecs.tsv and meta.tsv)

## Visualization

The program includes functions to plot:
- Training and validation accuracy
- Training and validation loss

## Testing

You can test the model with custom sentences:
```python
sentence = ["your text here"]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length)
prediction = model.predict(padded)
```

## Output Files

- vecs.tsv: Contains embedding vectors
- meta.tsv: Contains words corresponding to the vectors

## Note

This implementation is designed to run in Google Colab and includes Colab-specific features like file upload widgets and download capabilities.
