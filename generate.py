import numpy as np
import tweepy
from keras.models import model_from_yaml
import time


auth=tweepy.OAuthHandler("","")
auth.set_access_token("","")
api=tweepy.API(auth)
with open("lspd.txt", "r") as f:
	corpus = f.read()

chars = sorted(list(set(corpus)))
encoding = {c: i for i, c in enumerate(chars)}
decoding = {i: c for i, c in enumerate(chars)}
print("loaded encoding and decoding data sets")

sentence_length=60
with open ("model.yaml",'r') as f:
    yaml_string=f.read()

model=model_from_yaml(yaml_string)
model.load_weights("weights-30-0.890.hdf5")
seed_starting_index = np.random.randint(0, len(corpus) - sentence_length)
seed_sentence = corpus[seed_starting_index:seed_starting_index + sentence_length]
X_predict = np.zeros((1, sentence_length, len(chars)), dtype=np.bool)
for i, char in enumerate(seed_sentence):
    X_predict[0, i, encoding[char]] = 1


generated = ""
for i in range(170):
    prediction = np.argmax(model.predict(X_predict))
    generated += decoding[prediction]
    activations = np.zeros((1, 1, len(chars)), dtype=np.bool)
    activations[0, 0, prediction] = 1
    X_predict = np.concatenate((X_predict[:, 1:, :], activations), axis=1)

api.update_status(seed_sentence + generated)
print("Posted Tweet")
