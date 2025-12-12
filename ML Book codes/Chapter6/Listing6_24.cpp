#include "mbed.h"
#include "svm_cls_inference.h"
#include "lib_serial.h"
#include "lib_rng.h"
#include "lib_uart.h"
#include <string.h>

#define INPUT_PC 1
#define INPUT_MCU 2
#define INPUT INPUT_PC

#define SIZE_INPUT NUM_FEATURES
#define SIZE_OUTPUT ((NUM_CLASSES == 2) ? 1 : NUM_CLASSES)

float input[SIZE_INPUT];
float output[SIZE_OUTPUT];

int main()
{
    LIB_UART_Init();
#if (INPUT == INPUT_MCU)
    LIB_RNG_Init();
#endif
    while (true)
    {
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
}