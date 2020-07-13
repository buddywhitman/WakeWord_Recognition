from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Input,Conv1D,GRU,BatchNormalization,Dense,TimeDistributed, Activation, Dropout
from keras.optimizers import Adam
import numpy as np
import h5py


NUM_WAKE_WORDS=1

def model(input_shape):
    X_input = Input(shape = input_shape)
    X = Conv1D(48, kernel_size=10, strides=5)(X_input)
    X = BatchNormalization()(X)
    X = Activation('tanh')(X)
    X = Dropout(0.2)(X)

    X = GRU(units = 48, return_sequences = True, recurrent_activation='sigmoid')(X)
    X = Dropout(0.2)(X)
    X = BatchNormalization()(X)

    X = TimeDistributed(Dense(48, activation = "sigmoid"))(X)
    X= Dropout(0.2)(X)
    X = TimeDistributed(Dense(NUM_WAKE_WORDS+1, activation = "sigmoid"))(X)

    model = Model(inputs = X_input, outputs = X)
    return model

model=model(input_shape = (749,13))
AUDIO_LENGTH=7500.0
TRIGGER_OUTPUT_LENGTH=148
model.summary()
opt = Adam()
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
filepath="model_checkpoint.hdf5"
model.load_weights(filepath)
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True,save_weights_only=False, mode='auto',period=10)
callbacks_list = [checkpoint]

for n in range(1):
    print "Loading "+'dataset_'+str(n)+'.h5'
    h5f = h5py.File('dataset_'+str(n)+'.h5','r')
    samples = np.asarray(h5f['samples'])
    print samples.shape
    trigger_timings=np.asarray(h5f['trigger_timings'])
    h5f.close()
    labels=np.zeros((trigger_timings.shape[0],TRIGGER_OUTPUT_LENGTH,NUM_WAKE_WORDS+1))
    for i in range(trigger_timings.shape[0]):
            labels[i,:,NUM_WAKE_WORDS]=np.ones(TRIGGER_OUTPUT_LENGTH)
            if(trigger_timings[i,0]>=0):
                t=int((trigger_timings[i,1]/AUDIO_LENGTH)*TRIGGER_OUTPUT_LENGTH)
                labels[i,t:t+int((200/AUDIO_LENGTH)*TRIGGER_OUTPUT_LENGTH),trigger_timings[i,0]]=np.ones((int((200/AUDIO_LENGTH)*TRIGGER_OUTPUT_LENGTH)))
                labels[i,t:t+int((200/AUDIO_LENGTH)*TRIGGER_OUTPUT_LENGTH),NUM_WAKE_WORDS]=np.zeros((int((200/AUDIO_LENGTH)*TRIGGER_OUTPUT_LENGTH)))

    model.fit(samples, labels, batch_size = 1000, epochs=1000,validation_split=0.1,callbacks=callbacks_list)
