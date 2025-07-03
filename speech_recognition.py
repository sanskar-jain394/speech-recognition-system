from IPython.display import Audio 
from scipy.io import wavfile
import numpy as np 
  
file_name = 'sample_audio.wav' 

Audio(file_name)  
data = wavfile.read(file_name)
framerate = data[0]
sounddata = data[1]
time = np.arange(0,len(sounddata))/ framerate
print(framerate) 
print('total time:', len(sounddata)/framerate)

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import soundfile as sf
import librosa
file_path = "sample_audio.wav"
speech, sr = librosa.load(file_path, sr=16000)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
with torch.no_grad():
    logits = model(**inputs).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])

print("Transcription:", transcription)


