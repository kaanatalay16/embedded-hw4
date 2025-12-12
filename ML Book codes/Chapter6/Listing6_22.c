/* USER CODE BEGIN WHILE */
  while (1)
  {
/* USER CODE END WHILE */

/* USER CODE BEGIN 3 */
/*
* ORGANIZE INPUTS
*/
#if (INPUT == INPUT_PC)
 LIB_SERIAL_Receive(input, SIZE_INPUT, TYPE_F32);
#elif (INPUT == INPUT_MCU)
 for (uint32_t i = 0; i < SIZE_INPUT; ++i)
 input[i] = (float)(LIB_RNG_GetRandomNumber() % 1000) / 1000.0f;

 LIB_SERIAL_Transmit(input, SIZE_INPUT, TYPE_F32);
#endif
 svm_cls_score(input, output);
 LIB_SERIAL_Transmit((void*)&output, SIZE_OUTPUT, TYPE_F32);
 HAL_Delay(1000);
}
/* USER CODE END 3 */