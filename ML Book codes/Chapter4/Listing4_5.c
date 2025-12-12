/* USER CODE BEGIN Includes */
#include "lib_hts221.h"
#include "lib_serial.h"
/* USER CODE END Includes */

/* USER CODE BEGIN 0 */
float hts221[2] = {0};
/* USER CODE END 0 */

/* USER CODE BEGIN 2 */
LIB_HTS221_Init();
/* USER CODE END 2 */

/* USER CODE BEGIN WHILE */
while (1)
{
/* USER CODE END WHILE */

/* USER CODE BEGIN 3 */
LIB_HTS221_GetHumidity(&hts221[0]);
LIB_HTS221_GetTemperature(&hts221[1]);
LIB_SERIAL_Transmit(hts221, 2, TYPE_F32);
HAL_Delay(1000);
}
/* USER CODE END 3 */