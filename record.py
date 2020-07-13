import pyaudio
import wave
import time
import sys
from pydub import AudioSegment

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5.1

if(len(sys.argv)==2):
    WAVE_OUTPUT_FILENAME=str(sys.argv[1])+'.wav'
else:
    WAVE_OUTPUT_FILENAME = "test_record.wav"


audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
print "recording..."
frames = []

start_time=time.time()
while(time.time()-start_time<RECORD_SECONDS):
    data = stream.read(CHUNK)
    if(len(data)>0):
        # print data
        # print str([ord(c) for c in data])
        frames.append(data)
    time.sleep(0.01)
    print time.time()-start_time

print "finished recording"


# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()

waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
# print audio.get_sample_size(FORMAT)
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

sound=AudioSegment.from_wav(WAVE_OUTPUT_FILENAME)
sound=sound[0:5000]
sound.export(WAVE_OUTPUT_FILENAME,format="wav")
