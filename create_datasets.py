from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

import subprocess
import os
import shutil
from os import listdir
from os.path import isfile, join
from pydub import AudioSegment
from random import randint
import h5py
import numpy as np
from random import shuffle
import threading

DATASET_SIZE=8000
NUMBER_OF_DATASETS=1
AUDIO_LENGTH=7500 #ms
TRIGGER_LENGTH=200 #ms
PITCH_VARIATIONS=[0.9,1.0,1.1]
VOLUME_VARIATIONS=[-20,-10,0]

noise_file_list=[]
positive_file_list=[]
negative_file_list=[]
wake_words=[]

for path, subdirs, files in os.walk("Positives/"):
    for name in files:
        if(".wav" in name):
            wake_word_name=path.split("/")[1]
            if(wake_word_name not in wake_words):
                wake_words.append(wake_word_name)
            positive_file_list.append([wake_word_name,os.path.join(path, name)])
for path, subdirs, files in os.walk("Negatives/"):
    for name in files:
        if(".wav" in name):
            negative_file_list.append(os.path.join(path, name))

for path, subdirs, files in os.walk("Noise/"):
    for name in files:
        if(".wav" in name):
            noise_file_list.append(os.path.join(path, name))


shuffle(positive_file_list)
shuffle(negative_file_list)
shuffle(noise_file_list)

print("Wake words = "+ str(wake_words))
total_positive_files=len(positive_file_list)
total_negative_files=len(negative_file_list)
total_noise_files=len(noise_file_list)
print("Total positive files = {0}".format(total_positive_files))
print("Total negative files = {0}".format(total_negative_files))
print("Total noise files = {0}".format(total_noise_files))


current_positive_file_index=0
current_negative_file_index=0
current_noise_file_index=0


files_processed=0
datasets_saved=0
samples=[]
trigger_timings=[]
while(datasets_saved<NUMBER_OF_DATASETS):
    wake_word_name=positive_file_list[current_positive_file_index][0]
    positive_sound = AudioSegment.from_file(positive_file_list[current_positive_file_index][1], format="wav")
    negative_sound = AudioSegment.from_file(negative_file_list[current_negative_file_index], format="wav")
    current_positive_file_index+=1
    current_positive_file_index%=total_positive_files
    current_negative_file_index+=1
    current_negative_file_index%=total_negative_files
    utterances=[positive_sound,negative_sound]
    for u in range(2):
        for j in range(len(VOLUME_VARIATIONS)):
            for k in range(len(PITCH_VARIATIONS)):
                noise_sound = AudioSegment.from_file(noise_file_list[current_noise_file_index], format="wav")
                noise_length = len(noise_sound)
                current_noise_file_index+=1
                current_noise_file_index%=total_noise_files
                temp_sample_rate = int(PITCH_VARIATIONS[k]*utterances[u].frame_rate)
                temp_sound=utterances[u]._spawn(utterances[u].raw_data,overrides={'frame_rate': temp_sample_rate})
                temp_sound_length=len(temp_sound)
                noise_start_index=randint(0,noise_length-AUDIO_LENGTH)
                temp_noise=noise_sound[noise_start_index:noise_start_index+AUDIO_LENGTH]
                temp_noise=temp_noise+(temp_noise.dBFS-temp_sound.dBFS)+VOLUME_VARIATIONS[randint(0,len(VOLUME_VARIATIONS)-1)]
                overlay_position=randint(0,AUDIO_LENGTH-temp_sound_length-TRIGGER_LENGTH)
                export_sound=temp_noise.overlay(temp_sound,position=overlay_position)
                export_sound+=VOLUME_VARIATIONS[j]
                export_sound=export_sound.set_frame_rate(16000).set_channels(1)
                export_sound.export("temp_database_creation.wav",format="wav")
                (rate,sig) = wav.read("temp_database_creation.wav")
                mfcc_feat = np.asarray(mfcc(sig,rate))
                samples.append(mfcc_feat)
                if(u==0):
                    trigger_timings.append([wake_words.index(wake_word_name),overlay_position+temp_sound_length])
                else:
                    trigger_timings.append([-1,-1])

                files_processed+=1
                print("Files processed = {0}".format(files_processed))
                if(files_processed>=DATASET_SIZE):
                    samples=np.asarray(samples)
                    trigger_timings=np.asarray(trigger_timings)
                    h5f = h5py.File('dataset_'+str(datasets_saved)+'.h5', 'w')
                    h5f.create_dataset('samples', data=samples)
                    h5f.create_dataset('trigger_timings', data=trigger_timings)
                    h5f.close()
                    print "Saved : "+'dataset_'+str(datasets_saved)+'.h5'
                    samples=[]
                    trigger_timings=[]
                    files_processed=0
                    datasets_saved+=1
