#ifndef INC_MFCC_H_
#define INC_MFCC_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "stm32f746xx.h"
#include "arm_math.h"

#define SAMP_FREQ 8000
#define NUM_FBANK_BINS 40
#define MEL_LOW_FREQ 20
#define MEL_HIGH_FREQ 4000

#define M_2PI 6.283185307179586476925286766559005

struct MFCCInstance{
  const float *dct_matrix;
  float *windowCoefs;
  uint32_t *filterPos;
  uint32_t *filterLengths;
  uint32_t fftLen;
  uint32_t nbMelFilters;
  uint32_t nbDCtOutputs;
  arm_rfft_fast_instance_f32 *rfft;
};

typedef struct MFCCInstance mfcc_instance;

void init_mfcc_instance(mfcc_instance *S, uint32_t fftLen, uint32_t nbMelFilters, uint32_t nbDctOutputs);
float frequencyToMelSpace(float freq);
float melSpaceToFrequency(float mels);
float* create_dct_matrix(int numOfDctOutputs, int numOfMelFilters);
int create_mel_fbank(int FFTSize, int n_mels, uint32_t **filtPos, uint32_t **filtLen, float **packedFilters);
void mfcc_compute(const mfcc_instance S, const int16_t *audio_data, int frame_len, float *mfcc_out);

#endif /* INC_MFCC_H_ */
