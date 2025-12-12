/* USER CODE BEGIN Includes */
#include "mfcc.h"
#include "data.h"
/* USER CODE END Includes */

/* USER CODE BEGIN 2 */
mfcc_instance S;
float **mfcc_out;
int frame_len = 1024;
init_mfcc_instance(S, frame_len, 20, 13);
mfcc_compute(S, audio_data, frame_len, &mfcc_out);
/* USER CODE END 2 */

