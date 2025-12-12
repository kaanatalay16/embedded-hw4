#include "svm_moments_config.h"
const float coeffs[NUM_CLASSES - 1][NUM_SV] = {{ 0. , 0. , ...}, ...};
const float SV[NUM_SV][NUM_FEATURES] = {{ 3.15816167e-01, 1.75105811e-02, 5.41035080e-04, 2.53788016e-04,}, ...};
const float intercepts[NUM_INTERCEPTS] = {-6.33389052e-01, 5.45609269e-01, 2.04965002e+00, 6.64191770e-01,...};
const float w_sum[NUM_CLASSES + 1] = {0, 5007, 6097,12055,18186,24028,29445,35352,39414,44821,50770};
const float svm_gamma = 9.659468371895358;
const float coef0 = 0.0;
const int degree = 3;
const enum KernelType type = RBF;
