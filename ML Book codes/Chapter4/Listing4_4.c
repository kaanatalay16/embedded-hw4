/* USER CODE BEGIN Includes */
#include "lib_hts221.h"
/* USER CODE END Includes */

/* USER CODE BEGIN 0 */
float temperature;
float humidity;
/* USER CODE END 0 */

/* USER CODE BEGIN 2 */
LIB_HTS221_Init();
/* USER CODE END 2 */

/* USER CODE BEGIN WHILE */
while (1)
{
/* USER CODE END WHILE */

/* USER CODE BEGIN 3 */
  LIB_HTS221_GetHumidity(&humidity);
  LIB_HTS221_GetTemperature(&temperature);
  HAL_Delay(10);
}
/* USER CODE END 3 */
