import sounddevice as sd
import matplotlib.pyplot as plt 
import numpy as np

import librosa
from scipy import signal
import random
from scipy import misc
import tensorflow as tf

#LOAD PRETAINED MODEL
model = tf.keras.models.load_model('./model_words_recognizer')

# START CONFIGURATION
time = 1
sample_rate = 22050
records_in_screen = 2

samples = int(time * sample_rate)

window_size = 10
low_pass_hann_size = 10

# PLOT COFIGURATIONS
plt.ion()

fig = plt.figure()
ax = fig.add_subplot(211)
ax_spec = fig.add_subplot(212)



lag_signal = [0] * samples * records_in_screen
lag_envelope = lag_signal

#print(len(lag_signal))


line1, = ax.plot(lag_signal, 'b-') 
print("lag:{}".format(len(lag_signal)))


low_pass = np.convolve(lag_signal,np.hanning(low_pass_hann_size))/low_pass_hann_size
envelope = 15*np.convolve(np.abs(low_pass),[1]*window_size)/window_size

line2, = ax.plot(np.abs(lag_signal), 'g-')
print("sh_env:{}".format(np.shape(lag_signal)))
print("env:{}".format(88209))

ax.set(ylim=(-1, 1))



def preprocess_audio(audio):
	audio = audio/max(audio)
	m=librosa.feature.mfcc(y=audio,sr=sample_rate,n_mfcc=20)
	return m

def predict(spec):

	one_hot_prediction = model.predict(np.reshape(np.array([spec]),(1,20,44,1)))
	prediction = np.argmax(one_hot_prediction[0])
	dirs =  {
		0:	"left",
		1:	"down",
		2:	"right",
		3:	"up"
	}
	print(dirs[prediction])

finding_end = 0 

new_recording = sd.rec(samples, samplerate=sample_rate , channels=1)
sd.wait()

while(True): #30 updates in 1 second
	temp_recording = sd.rec(samples, samplerate=sample_rate , channels=1)
	#sd.wait()
	#Update signal

	lag_signal[:(records_in_screen - 1)*samples] = lag_signal[-(records_in_screen - 1)*samples:]
	lag_signal[-samples:] = np.reshape(new_recording,(len(new_recording)))
	
	line1.set_ydata(lag_signal)
	
	#Update activatio intensity	

	#low_pass = 15*np.abs(np.convolve(lag_signal,np.hanning(low_pass_hann_size)))/low_pass_hann_size
	intensity = 3 * np.abs(lag_signal) - .6

	line2.set_ydata(intensity)
	intensity_umbral = intensity[-samples:] > 0

	
	if(np.amax(intensity[-samples:])>0):

		print("_________Voz___________")

		ax_spec.imshow(preprocess_audio(lag_signal[-samples:]))
		
		finding_end = 1
	if(finding_end == 1):
		subdivitions = 5
		for i in range(subdivitions):
			if(intensity_umbral[ int(samples * i / subdivitions):int(samples * (i+1) / subdivitions)].any() == False):
				end_point = int(samples * i / subdivitions) + samples * (records_in_screen - 1)
				start_point = end_point - samples
				#ax_spec.plot(lag_signal[start_point:end_point])
				spec = preprocess_audio(lag_signal[start_point:end_point])
				ax_spec.imshow(spec)
				predict(spec)
				finding_end = 0
				

	fig.canvas.draw()
	fig.canvas.flush_events()
	sd.wait()
	new_recording = temp_recording
