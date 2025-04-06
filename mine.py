ACTIVATION FUNCTIONS :

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize the datasets
x_train, x_test = x_train / 255.0, x_test / 255.0
# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'), # Try changing the neuron count & activation function
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Train the model
model.fit(x_train, y_train, epochs=8) # try incresing the epochs
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("\nSummary of results:")
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

#similarly change the neurons and activation function for the hidden layer and plot a graph for accuracy
import matplotlib.pyplot as plt


# putting the results in list
neuroacti = ["sigmoid 64", "sigmoid 128", "relu 64", "relu 128"]
acc=[97.68,98.55,97.56,98.44]

# Create a bar graph
plt.bar(neuroacti, acc)

# Add titles and labels
plt.title('Accuracy of models')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(97.0, 100.0)

# Display the bar graph
plt.show()



XOR :

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# XOR input data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# XOR output data
y = np.array([[0], [1], [1], [0]])

model = Sequential()
# Input layer with 2 neurons and hidden layer with 2 neurons
model.add(Dense(6, input_dim=2, activation='relu'))
# Output layer with 1 neuron
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=1000, verbose=0)

loss, accuracy = model.evaluate(X, y)
print(f'Accuracy: {accuracy * 100}%')

predictions = model.predict(X)
predictions_int = [round(pred[0]) for pred in predictions]
print('Predictions:',predictions_int)



SPEECH :

!pip install speechrecognition pydub tensorflow numpy
!pip install --upgrade --force-reinstall pandas

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import os
import numpy as np
from pydub import AudioSegment
import speech_recognition as sr


# Step 1: Convert MP3 to WAV
def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

# Step 2: Use SpeechRecognition to get text from audio
def transcribe_audio_google(wav_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            print("Transcribed:", text)
            return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            print(f"API error: {e}")
            return ""

# Step 3: Preprocess label for training
def prepare_dataset(transcription):
    label = 1 if "hello" in transcription.lower() else 0
    return np.array([label])

# Step 4: Dummy audio features (for demo purposes)
def extract_features(wav_path):
    # Normally, you'd extract MFCC or spectrogram features
    # Here we just use a dummy fixed-length array
    return np.random.rand(20)

# Step 5: Train a simple neural network
def train_model(X, y):
    model = Sequential([
        Dense(16, input_shape=(20,), activation='relu'),
        Dense(8, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, to_categorical(y), epochs=10, verbose=1)
    return model

# === Main Flow ===
mp3_file = "/content/drive/MyDrive/dl_lab/hello.mp3"
wav_file = "converted_audio.wav"

convert_mp3_to_wav(mp3_file, wav_file)
text = transcribe_audio_google(wav_file)
y = prepare_dataset(text)
X = np.array([extract_features(wav_file)])

model = train_model(X, y)

# Prediction
pred = model.predict(X)
print("Prediction:", "hello" if np.argmax(pred) == 1 else "not hello")



TRAFFIC SIGN :

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.metrics import accuracy_score
print(1)
# Define dataset path
dataset_path = r"C:\\Users\SAKTHI M\\Downloads\\exp6"
# Load training data
data = []
labels = []
classes = 43
print(2)
for i in range(classes):
    path = os.path.join(dataset_path, 'Train', str(i))
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(os.path.join(path, a))
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print(f"Error loading image {a} in class {i}")
print(3)
# Convert lists into numpy arrays
data = np.array(data)
labels = np.array(labels)
# Display dataset shape
print("Dataset shape:", data.shape, labels.shape)
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)
print(4)
# Define CNN model
model = Sequential([
    Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]),
    Conv2D(filters=32, kernel_size=(5, 5), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(rate=0.25),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(rate=0.25),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(rate=0.5),
    Dense(43, activation='softmax')
])
print(5)
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Display model summary
model.summary()
# Train the model
epochs = 15
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
# Plot training accuracy and loss
plt.figure(0)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Load test dataset
y_test_df = pd.read_csv(os.path.join(dataset_path, 'Test.csv'))
labels = y_test_df["ClassId"].values
imgs = y_test_df["Path"].values
data = []
for img in imgs:
    image = Image.open(os.path.join(dataset_path, img))
    image = image.resize((30, 30))
    data.append(np.array(image))
X_test = np.array(data)
# Make predictions
pred = np.argmax(model.predict(X_test), axis=-1)
# Print accuracy score
print("Test Accuracy:", accuracy_score(labels, pred))


ANN:

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert labels to one-hot encoded format
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
# Flatten the input image
model.add(Flatten(input_shape=(28, 28)))
# Hidden layer with 128 neurons and ReLU activation
model.add(Dense(128, activation='relu'))
# Output layer with 10 neurons (one for each digit) and softmax activation
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100}%')

predictions = model.predict(X_test)
# Convert predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)
print('Predicted Classes:',predicted_classes)

#Save the model

model.save("mnist_ffnn_model.h5")

# Upload and validate with a custom hand-drawn image

print("Upload a hand-drawn digit image (28x28 grayscale, or it will be resized):")

uploaded files.upload()

for filename in uploaded.keys():

#Load the uploaded image
img Image.open(filename).convert('L') # Convert to grayscale
img ImageOps.invert(img)

#Invert the colors
img img.resize((28, 28))

#Resize to 28x28
# Display the uploaded image
plt.imshow(img, cmap='gray')
plt.title("Uploaded Image")
plt.axis('off')
plt.show()

#Preprocess the image
img_array = np.array(img)/255.0

#Normalize pixel values
img_array = img_array.reshape(1, 28, 28) # Reshape for model input

#Make a prediction

prediction model.predict(img_array)
predicted_label = np.argmax(prediction)
print(f'Predicted Digit: (predicted_label}")

TWITTER :

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import spacy

columns = ['id','country','Label','Text']
df = pd.read_csv("/content/drive/MyDrive/twitter_training.csv", names=columns)

print(df.shape)
df.head(5)

for i in range(5):
    print(f"{i+1}: {df['Text'][i]} -> {df['Label'][i]}")

df.dropna(inplace=True)

nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    return " ".join(filtered_tokens)

df['Preprocessed Text'] = df['Text'].apply(preprocess)

le_model = LabelEncoder()
df['Label'] = le_model.fit_transform(df['Label'])

X_train, X_test, y_train, y_test = train_test_split(df['Preprocessed Text'], df['Label'],
                                                    test_size=0.2, random_state=42, stratify=df['Label'])

print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)

clf = Pipeline([
    ('vectorizer_tri_grams', TfidfVectorizer()),
    ('naive_bayes', MultinomialNB())
])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

clf = Pipeline([
    ('vectorizer_tri_grams', TfidfVectorizer()),
    ('naive_bayes', RandomForestClassifier())
])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

test_df = pd.read_csv('/content/drive/MyDrive/twitter_validation.csv', names=columns)
test_text = test_df['Text'][10]
print(f"{test_text} ===> {test_df['Label'][10]}")

test_text_processed = [preprocess(test_text)]
test_text = clf.predict(test_text_processed)

classes = ['Irrelevant', 'Natural', 'Negative', 'Positive']
print(f"True Label: {test_df['Label'][10]}")
print(f'Predict Label: {classes[test_text[0]]}')



