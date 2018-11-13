# Laxmi Prasad Devkota Poem Generator

A machine learning LSTM model trained on a 100k character set of Nepali poet Laxmi Prasad Devkota. Trained through 30 epochs with a loss of 0.890. Generates 170 devnagari characters based on 60 characters taken from actual poems and posts them on twitter every hour. [Twitter](https://twitter.com/lspd_bot)

## Files

##### generate.py
>Loads weight and model architecure, generates the text and posts it in Twitte
##### train.py
>Contains code to train a new LSTM model using keras. Requires a corpus containing the original text.
##### lspd.txt
>The dataset used to train the model. In this case, the file contains a collection of poems of Laxmi Prasad Devkota and contains 119161 characters. (97307 with no spaces)
##### model.yaml
>Model architecture written in yaml using keras' model.to_yaml() function
##### weights-30-0.890.hdf5
>Pretrained model weights used for generating text. Taken after 30 epochs of training with a loss of 0.890
