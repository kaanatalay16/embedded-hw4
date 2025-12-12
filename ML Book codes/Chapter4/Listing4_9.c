/* USER CODE BEGIN Includes */
#include "lib_bno055.h"
#include "lib_serial.h"
/* USER CODE END Includes */

/* USER CODE BEGIN 0 */
BNO055_F32DataTypeDef bno055Data;
/* USER CODE END 0 */

/* USER CODE BEGIN 2 */
LIB_BNO055_Init();
/* USER CODE END 2 */

/* USER CODE BEGIN WHILE */
while (1)
{
/* USER CODE END WHILE */

/* USER CODE BEGIN 3 */
  LIB_BNO055_ReadAccelXYZ(&bno055Data.accel[0], &bno055Data.accel[1], &bno055Data.accel[2]);
  LIB_BNO055_ReadGyroXYZ(&bno055Data.gyro[0], &bno055Data.gyro[1], &bno055Data.gyro[2]);
  LIB_BNO055_ReadMagXYZ(&bno055Data.mag[0], &bno055Data.mag[1], &bno055Data.mag[2]);
  LIB_SERIAL_Transmit(&bno055Data, 9, TYPE_F32);
  HAL_Delay(1000);
}
/* USER CODE END 3 */