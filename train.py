import numpy as np
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import Adam

with open ("lspd.txt","r") as f:
  corpus=f.read()
  
  
char=sorted(list(set(corpus)))

encoding={c:i for i,c in enumerate(char)}
decoding={i:c for i,c in enumerate(char)}

print(encoding)
print(decoding)
print("Unique characters "+str(len(char)))
print("Total characters "+str(len(corpus)))

sentence_length=60
step=1
sentences=[]
nextchar=[]

for i in range (0,len(corpus)-sentence_length,step):
  sentences.append(corpus[i:i+sentence_length])
  nextchar.append(corpus[i+sentence_length])
  
  
print("Train set length "+str(len(nextchar)))


x=np.zeros((len(sentences),sentence_length,len(char)),dtype=np.bool)
y=np.zeros((len(sentences),len(char)),dtype=np.bool)

for i,sentence in enumerate(sentences):
  for t,character in enumerate(sentence):
    x[i,t,encoding[character]]=1
  y[i,encoding[nextchar[i]]]=1
  
print("Training set(X) shape"+str(x.shape))
print("Training set(Y) shape"+str(y.shape))

model=Sequential()
model.add(LSTM(256,input_shape=(sentence_length,len(char))))
model.add(Dense(len(char)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
architecture = model.to_yaml()
with open('model.yaml', 'a') as model_file:
    model_file.write(architecture)

file_path="weights-{epoch:02d}-{loss:.3f}.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor="loss", verbose=1, save_best_only=True, mode="min")
callbacks = [checkpoint]

model.fit(x,y,nb_epoch=30,batch_size=128,callbacks=callbacks)
