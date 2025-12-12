#include "mfcc.h"

void init_mfcc_instance(mfcc_instance *S, uint32_t fftLen,
		uint32_t nbMelFilters, uint32_t nbDctOutputs) {

	//create window function
	float *window_func = malloc(sizeof(float) * fftLen);
	for (int i = 0; i < fftLen; i++)
		window_func[i] = 0.5 - 0.5 * cos(M_2PI * ((float) i) / (fftLen));

	//create mel filterbank
	S->filterPos = malloc(nbMelFilters * sizeof(int));
	S->filterLengths = malloc(nbMelFilters * sizeof(int));
	S->nbMelFilters = nbMelFilters;
	S->nbDCtOutputs = nbDctOutputs;
	S->windowCoefs = window_func;

	create_mel_fbank(fftLen, nbMelFilters, &S->filterPos, &S->filterLengths, &S->windowCoefs);

	//create DCT matrix
	S->dct_matrix = create_dct_matrix(nbMelFilters, nbDctOutputs);

	//initialize FFT
//	  S->rfft = arm_rfft_fast_instance_f32;
	arm_rfft_fast_init_f32(S->rfft, fftLen);
}

float frequencyToMelSpace(float freq) {
	return 1127.0 * logf(1.0 + freq / 700.0);
}

float melSpaceToFrequency(float mel) {
	return 700.0 * (expf(mel / 1127.0) - 1.0);
}

float* create_dct_matrix(const int numOfDctOutputs, const int numOfMelFilters) {
	float *dct_matrix = malloc(
			sizeof(float) * numOfDctOutputs * numOfMelFilters);
	float norm_mels = sqrt(2.0 / numOfMelFilters);
	for (int mel_idx = 0; mel_idx < numOfMelFilters; mel_idx++) {
		for (int dct_idx = 0; dct_idx < numOfDctOutputs; dct_idx++) {
			float s = (mel_idx + 0.5) / numOfMelFilters;
			dct_matrix[dct_idx * numOfMelFilters + mel_idx] = (cosf(
					dct_idx * M_PI * s) * norm_mels);
		}
	}

	return dct_matrix;
}

int create_mel_fbank(int FFTSize, int n_mels, uint32_t **filtPos,
		uint32_t **filtLen, float **packedFilters) {

	int half_fft_size = FFTSize / 2;
	float filters[n_mels][half_fft_size + 1];
	float spectrogram_mel[half_fft_size];

	float fmin_mel = frequencyToMelSpace(MEL_LOW_FREQ);
	float fmax_mel = frequencyToMelSpace(MEL_HIGH_FREQ);
	float freq_step = SAMP_FREQ / FFTSize;

	for (int freq_idx = 1; freq_idx < half_fft_size + 1; freq_idx++) {
		float linear_freq = freq_idx * freq_step;
		spectrogram_mel[freq_idx - 1] = frequencyToMelSpace(linear_freq);
	}

	float mel_step = (fmax_mel - fmin_mel) / (n_mels + 1);
	int totalLen = 0;
	for (int mel_idx = 0; mel_idx < n_mels; mel_idx++) {
		float mel = mel_step * mel_idx;
		bool startFound = false;
		int startPos = 0, endPos = 0, curLen = 0;
		for (int freq_idx = 0; freq_idx < half_fft_size; freq_idx++) {
			float upper = (spectrogram_mel[freq_idx] - mel) / mel_step;
			float lower = (mel - spectrogram_mel[freq_idx]) / mel_step + 2;
			float filter_val = fmaxf(0.0, fminf(upper, lower));
			filters[mel_idx][freq_idx + 1] = filter_val;
			if (!startFound & (filter_val != 0.0)) {
				startFound = true;
				startPos = freq_idx + 1;
			}

			else if (startFound & (filter_val == 0.0)) {
				endPos = freq_idx;
				break;
			}
		}
		curLen = endPos - startPos + 1;
		*filtLen[mel_idx] = (endPos - startPos + 1);
		*filtPos[mel_idx] = startPos;
		*packedFilters = realloc(*packedFilters,
				(totalLen + curLen) * sizeof(float));
		if (*packedFilters == NULL) {
			printf("Memory allocation failed\n");
			return 1;
		}

		memcpy(*packedFilters + totalLen, &filters[mel_idx][startPos],
				curLen * sizeof(float));
		totalLen += curLen;
	}

	return 0;
}

void mfcc_compute(const mfcc_instance S, const int16_t *audio_data, int frame_len, float *mfcc_out) {

	int32_t i, j, bin;
	float frame[frame_len];
	float buffer[frame_len];
	float mel_energies[S.nbMelFilters];

	for (i = 0; i < frame_len; i++) {
		frame[i] = (float) audio_data[i] / (1 << 15);
	}
	//Fill up remaining with zeros

	for (i = 0; i < frame_len; i++) {
		frame[i] *= S->windowCoefs[i];
	}

	//Compute FFT
	arm_rfft_fast_f32(S.rfft, frame, buffer, 0);

	//Convert to power spectrum
	//frame is stored as [real0, realN/2-1, real1, im1, real2, im2, ...]
	int32_t half_dim = frame_len / 2;
	float first_energy = buffer[0] * buffer[0], last_energy = buffer[1]
			* buffer[1];  // handle this special case
	for (i = 1; i < half_dim; i++) {
		float real = buffer[i * 2], im = buffer[i * 2 + 1];
		buffer[i] = real * real + im * im;
	}
	buffer[0] = first_energy;
	buffer[half_dim] = last_energy;

	float sqrt_data;
	//Apply mel filterbanks
	for (bin = 0; bin < S.nbMelFilters; bin++) {
		j = 0;
		float mel_energy = 0;
		int32_t first_index = S.filterPos[bin];
		int32_t length = S.filterLengths[bin];
		for (i = first_index; i <= first_index + length; i++) {
			arm_sqrt_f32(buffer[i], &sqrt_data);
			mel_energy += (sqrt_data) * S.windowCoefs[bin * first_index + j];
			j++;
		}
		mel_energies[bin] = mel_energy;

		//avoid log of zero
		if (mel_energy == 0.0)
			mel_energies[bin] = 1e-7f;
	}

	//Take log
	for (bin = 0; bin < S.nbMelFilters; bin++)
		mel_energies[bin] = logf(mel_energies[bin]);

	//Take DCT. Uses matrix mul.
	for (i = 0; i < S.nbDCtOutputs; i++) {
		float sum = 0.0;
		for (j = 0; j < S.nbMelFilters; j++) {
			sum += S.dct_matrix[i * S.nbMelFilters + j] * mel_energies[j];
		}
		mfcc_out[i] = sum;
	}

}
