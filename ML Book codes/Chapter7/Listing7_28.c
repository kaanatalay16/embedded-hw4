#include "dt_reg_config.h"
const int LEFT_CHILDREN[NUM_NODES] = {  1,  2,  3,..., 397, -1, -1};
const int RIGHT_CHILDREN[NUM_NODES] = { 72, 13, 10,..., 398, -1, -1} ;
const int SPLIT_FEATURE[NUM_NODES] = { 0, 0, 0,..., 0,-2,-2};
const float THRESHOLDS[NUM_NODES] = { 1.77499998, 0.27500001, 0.175,..., 9.92499971,-2. ,-2.};
const float VALUES[NUM_NODES] = { 0.14443214, 0.7366521 , 0.21800425,..., 0.22977443, 0.1335265, 0.32602236};