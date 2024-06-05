from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input, TimeDistributed
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import pickle

# Set the path to the directory containing the images and captions
image_dir = 'images'
captions_file = 'captions.txt'

# Load the captions
with open(captions_file, 'r') as f:
  captions = f.readlines()

# Preprocess the captions
captions = [caption.strip().split() for caption in captions]

# Create a vocabulary of all the words in the captions
vocabulary = set()
for caption in captions:
  for word in caption:
    vocabulary.add(word)

# Create a mapping from word to index
word_to_index = {word: index for index, word in enumerate(vocabulary)}

# Define the maximum length of a caption
max_length = max([len(caption) for caption in captions])

# Load the pre-trained image recognition model
model = VGG16(weights='imagenet', include_top=False)

# Extract features from the images
image_features = []
for image_file in os.listdir(image_dir):
  image_path = os.path.join(image_dir, image_file)
  img = image.load_img(image_path, target_size=(224, 224))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  features = model.predict(x)
  features = features.flatten()
  image_features.append(features)

# Create the image captioning model
input_image = Input(shape=(image_features[0].shape[0],))
encoded_image = Dense(256, activation='relu')(input_image)
encoded_image = Dropout(0.5)(encoded_image)
input_caption = Input(shape=(max_length,))
embedded_caption = Embedding(len(vocabulary), 256)(input_caption)
lstm = LSTM(256, return_sequences=True)(embedded_caption)
merged = concatenate([encoded_image, lstm], axis=1)
output_caption = TimeDistributed(Dense(len(vocabulary), activation='softmax'))(merged)
model = Model(inputs=[input_image, input_caption], outputs=output_caption)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001))

# Train the model
for epoch in range(10):
  for i, caption in enumerate(captions):
    # Encode the caption
    encoded_caption = np.zeros((1, max_length))
    for j, word in enumerate(caption):
      encoded_caption[0, j] = word_to_index[word]

    # Train the model
    model.fit([image_features[i], encoded_caption], np.zeros((1, max_length, len(vocabulary))), epochs=1, verbose=0)

# Save the model
model.save('image_captioning_model.h5')

# Save the vocabulary and word-to-index mapping
with open('vocabulary.pkl', 'wb') as f:
  pickle.dump(vocabulary, f)
with open('word_to_index.pkl', 'wb') as f:
  pickle.dump(word_to_index, f)