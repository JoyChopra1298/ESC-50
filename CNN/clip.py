import numpy as np
import pydub
import librosa
import scipy
import scipy.fftpack as fft

silence_threshold = 60		# in -dB relative to max sound which is 0dB
lambdaa = 1				# amplitude of delta signal in PEFBEs
n_mels = 60 				# feature dimension for each frame
segment_length = 41			# 1 segment is 41 frames
segment_hop_length = 20		# nearly 50% overlap

class Clip:
	"""A single 5-sec long recording."""
	
	RATE = 22050   			# All recordings in ESC are 44.1 kHz but the paper downsampled to 22.05kHz
	frame_length=550 		# 25 ms windows
	hop_length=275 			# 50% overlap

	class Audio:
		"""The actual audio data of the clip.
		
			Uses a context manager to load/unload the raw audio data. This way clips
			can be processed sequentially with reasonable memory usage.
		"""
		
		def __init__(self, path):
			self.path = path
		
		def __enter__(self):
			# Actual recordings are sometimes not frame accurate, so we trim/overlay to exactly 5 seconds
			self.data = pydub.AudioSegment.silent(duration=5000)
			self.data = self.data.overlay((pydub.AudioSegment.from_file(self.path)[0:5000]).set_frame_rate(Clip.RATE))
			self.raw = (np.fromstring(self.data._data, dtype="int16") + 0.5) / (0x7FFF + 0.5)   # convert to float
			return(self)
		
		def __exit__(self, exception_type, exception_value, traceback):
			if exception_type is not None:
				print (exception_type, exception_value, traceback)
			del self.data
			del self.raw

	def __init__(self, audiopath,path):
		self.path = path
		self.target = (self.path.split(".")[0]).split("-")[-1]
		self.fold = self.path.split("-")[0]
		self.audio = Clip.Audio(audiopath+"/"+self.path)
		self.category = None

		with self.audio as audio:
			self.is_silent = librosa.effects._signal_to_frame_nonsilent(audio.raw,top_db=silence_threshold,frame_length=Clip.frame_length, hop_length=Clip.hop_length)
			self.non_silent = self.remove_silence(audio)
			################# Unsegmented features. 60 - dimensional ###################
			self.compute_PEFBEs()
			self.compute_FBEs()
			self.num_frames = len(self.non_silent)
			del self.is_silent
			del self.non_silent
			
		######################## Segment the clip into smaller parts. 41 frames(50% overlap) in the PEFBE paper. ########################
		self.mel_spectra = self.segment(self.mel_spectra.T).T
		self.log_spectra = self.segment(self.log_spectra.T).T
		self.log_delta = self.segment(self.log_delta.T).T
		self.log_delta2 = self.segment(self.log_delta2.T).T

		self.PEmel_spectra = self.segment(self.PEmel_spectra.T).T
		self.PElog_spectra = self.segment(self.PElog_spectra.T).T
		self.PElog_delta = self.segment(self.PElog_delta.T).T
		self.PElog_delta2 = self.segment(self.PElog_delta2.T).T

	def remove_silence(self,audio):
		# returns a list of numpy arrays (list of frames)
		newsig = []
		j = 0
		while j < len(self.is_silent):
			silent_count = 0
			#look for continuous silent frames
			while(j<len(self.is_silent) and (not self.is_silent[j])):
				silent_count +=1
				j+=1

			#skip all these frames if more than 3 continuously
			if(silent_count<=3):
				if(silent_count==0):
					newsig.append(audio.raw[(j)*Clip.hop_length:(j+2)*Clip.hop_length])
				for k in range(silent_count):
					newsig.append(audio.raw[(j+k)*Clip.hop_length:(j+k+2)*Clip.hop_length])
				j += silent_count
			j+=1

		#drop the partially filled frames 
		while(len(newsig[-1])!=Clip.frame_length):
			del(newsig[-1])
		newsig.append(audio.raw[-Clip.frame_length:])
		return newsig
	
	def compute_PEFBEs(self):
		power_spectra = []
		for frame in self.non_silent:
			delta = lambdaa*scipy.signal.unit_impulse(Clip.frame_length)
			frame += delta
			fft_frame = fft.fft(frame)
			normalised_frame = (fft_frame - np.mean(fft_frame)) / np.std(fft_frame)
			power_frame = np.abs(fft_frame)**2
			power_spectra.append(power_frame)	
		power_spectra = np.array(power_spectra)
		self.PEmel_spectra = librosa.feature.melspectrogram(S=power_spectra.T,n_mels=n_mels)
		self.PElog_spectra = librosa.core.power_to_db(self.PEmel_spectra)
		self.PElog_delta = librosa.feature.delta(self.PElog_spectra)
		self.PElog_delta2 = librosa.feature.delta(self.PElog_delta)

	def compute_FBEs(self):
		power_spectra = []
		for frame in self.non_silent:
			fft_frame = fft.fft(frame)
			power_frame = np.abs(fft_frame)**2
			power_spectra.append(power_frame)	
		power_spectra = np.array(power_spectra)
		self.mel_spectra = librosa.feature.melspectrogram(S=power_spectra.T,n_mels=n_mels)
		self.log_spectra = librosa.core.power_to_db(self.mel_spectra)
		self.log_delta = librosa.feature.delta(self.log_spectra)
		self.log_delta2 = librosa.feature.delta(self.log_delta)

	def segment(self,list):
		newsig = []
		n = len(list)
		if(n < segment_length):
			#### Make a segment by duplicating frames 
			new_segment = []
			for j in range(int(segment_length/n)):
				new_segment.extend(list[:])
			new_segment.extend(list[:segment_length - n])
			newsig.append(np.array(new_segment))	
		else:	
			for j in range(int(n/segment_hop_length)):
				newsig.append(list[j*segment_hop_length:(j+2)*segment_hop_length+1])

			#remove partially-filled segments from the end
			while(len(newsig[-1])!=segment_length):
				del(newsig[-1])
			# add a segment for last few frames tht might have been left out
			if(len(list)%segment_length != 0):
				newsig.append(list[-segment_length:])

		return np.array(newsig)
	
	def _print_stats(self,data):
		print(data.shape,np.max(data),np.min(data),np.mean(data),np.std(data))

	def print_clip_stats(self):
		print("length max min mean std")
		print("FBE mel ----------------------------------")
		self._print_stats(self.mel_spectra)
		print("FBE log ------------------------------")
		self._print_stats(self.log_spectra)
		print("FBE log delta ------------------------------")
		self._print_stats(self.log_delta)
		print("FBE log delta2 ------------------------------")
		self._print_stats(self.log_delta2)
		print("PEFBE mel ----------------------------------")
		self._print_stats(self.PEmel_spectra)
		print("PEFBE log ------------------------------")
		self._print_stats(self.PElog_spectra)
		print("PEFBE log delta------------------------------")
		self._print_stats(self.PElog_delta)
		print("PEFBE log delta2 ------------------------------")
		self._print_stats(self.PElog_delta2)
		print(len(self.non_silent))	

	def __repr__(self):
		return '<Target:{0}|Category:{1}|Fold:{2}|Number of frames:{3}|Number of segments:{4}>\nClip name : {5}'.format(self.target,self.category,self.fold,self.num_frames,self.log_spectra.shape[2],self.path)