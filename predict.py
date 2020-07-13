from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input,Conv1D,GRU,BatchNormalization,Dense,TimeDistributed, Activation, Dropout
from keras.optimizers import Adam
import numpy as np
import h5py
import matplotlib.pyplot as plt
from python_speech_features import mfcc
import scipy.io.wavfile as wav


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

(rate,sig) = wav.read("test_record.wav")
mfcc_feat = np.asarray(mfcc(sig,rate))
S=np.asarray([mfcc_feat])
prediction=model.predict(S)
prediction=np.squeeze(prediction)
plt.plot(prediction)
plt.show()
