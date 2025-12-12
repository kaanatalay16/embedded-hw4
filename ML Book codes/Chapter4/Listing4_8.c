/* USER CODE BEGIN Includes */
#include "lib_bno055.h"
/* USER CODE END Includes */

/* USER CODE BEGIN 0 */
float accel[3];
float gyro[3];
float mag[3];
/* USER CODE END 0 */

/* USER CODE BEGIN 2 */
LIB_BNO055_Init();
/* USER CODE END 2 */

/* USER CODE BEGIN WHILE */
while (1)
{
/* USER CODE END WHILE */

/* USER CODE BEGIN 3 */
  LIB_BNO055_ReadAccelXYZ(&accel[0], &accel[1], &accel[2]);
  LIB_BNO055_ReadGyroXYZ(&gyro[0], &gyro[1], &gyro[2]);
  LIB_BNO055_ReadMagXYZ(&mag[0], &mag[1], &mag[2]);
}
/* USER CODE END 3 */