import pyaudio
import collections
from struct import unpack
import time
import numpy as np
from python_speech_features import mfcc
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input,Conv1D,GRU,BatchNormalization,Dense,TimeDistributed, Activation, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import Queue

callback_queue = Queue.Queue()

NUM_WAKE_WORDS=1

def model(input_shape):
    X_input = Input(shape = input_shape)
    X = Conv1D(48, kernel_size=10, strides=5)(X_input)
    X = BatchNormalization()(X)
    X = Activation('tanh')(X)
    X = Dropout(0.4)(X)

    X = GRU(units = 48, return_sequences = True, recurrent_activation='sigmoid')(X)
    X = Dropout(0.4)(X)
    X = BatchNormalization()(X)

    X = TimeDistributed(Dense(48, activation = "sigmoid"))(X)
    X= Dropout(0.4)(X)
    X = TimeDistributed(Dense(NUM_WAKE_WORDS+1, activation = "sigmoid"))(X)

    model = Model(inputs = X_input, outputs = X)
    return model


model=model(input_shape = (None,13))
model.summary()
filepath="model_checkpoint.hdf5"
model.load_weights(filepath)

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = 800
MAX_KEYWORD_TIME=5#seconds
MAX_KEYWORD_LEN=int(MAX_KEYWORD_TIME*SAMPLE_RATE)

rb=collections.deque(maxlen=MAX_KEYWORD_LEN)
rb.extend([0]*(MAX_KEYWORD_LEN-1))

audio = pyaudio.PyAudio()

def audio_callback(in_data, frame_count, time_info, status):
    global rb
    datai=unpack('h'*(len(in_data)/2),in_data)
    rb.extend(datai)
    play_data = chr(0) * len(in_data)
    return play_data, pyaudio.paContinue

stream = audio.open(format=FORMAT, channels=CHANNELS,rate=SAMPLE_RATE, input=True,frames_per_buffer=CHUNK,stream_callback=audio_callback)



print "filling..."
time.sleep(5)
print "start talking"

while(True):
    sig=np.asarray(list(rb))
    sig=sig.astype('float64')
    mfcc_feat = np.asarray(mfcc(sig,16000))
    S=np.asarray([mfcc_feat])
    prediction=model.predict(S)
    prediction=np.squeeze(prediction)
    for p in prediction:
        if(p[0]>0.2):
            print("Triggered")
            rb.extend([0]*(MAX_KEYWORD_LEN-1))
            plt.plot(prediction)
            plt.show()
