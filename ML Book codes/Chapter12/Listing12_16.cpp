#include "mbed.h"
#include "network.h"
#include "lib_cube.h"
#include "lib_rng.h"
#include "lib_serial.h"
#include "lib_uart.h"

#define INPUT_PC 1
#define INPUT_MCU 2
#define INPUT INPUT_PC

#define SIZE_INPUT AI_NETWORK_IN_1_SIZE
#define SIZE_OUTPUT AI_NETWORK_OUT_1_SIZE

float input[SIZE_INPUT];
float output[SIZE_OUTPUT];

int main()
{
LIB_UART_Init();
LIB_CUBE_Init();
while (true)
{
#if (INPUT == INPUT_PC)
 LIB_SERIAL_Receive(input, SIZE_INPUT, TYPE_F32);
#elif (INPUT == INPUT_MCU)
 for (uint32_t i = 0; i < SIZE_INPUT; ++i)
 input[i] = (float)(LIB_RNG_GetRandomNumber() % 1000) / 1000.0f;
LIB_SERIAL_Transmit(input, SIZE_INPUT, TYPE_F32);
#endif
LIB_CUBE_Run(input, output);
LIB_SERIAL_Transmit(output, SIZE_OUTPUT, TYPE_F32);
HAL_Delay(1000);
}
}