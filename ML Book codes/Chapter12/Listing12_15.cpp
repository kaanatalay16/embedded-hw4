#include "mbed.h"
#include "classification.h"
#include "lib_uart.h"
#include "lib_serial.h"
#include "lib_rng.h"
#include "lib_model.h"

#define INPUT_PC 1
#define INPUT_MCU 2
#define INPUT INPUT_PC

#define SIZE_INPUT 2
#define SIZE_OUTPUT 1

constexpr int kTensorArenaSize = 136 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

int main()
{
LIB_MODEL_Init(converted_model_tflite, &input, tensor_arena, kTensorArenaSize);
LIB_UART_Init();
#if (INPUT == INPUT_MCU)
 LIB_RNG_Init();
#endif
while (true)
{
#if (INPUT == INPUT_PC)
 LIB_SERIAL_Receive(input->data.f, SIZE_INPUT, TYPE_F32);
#elif (INPUT == INPUT_MCU)
 for (uint32_t i = 0; i < SIZE_INPUT; ++i)
 input->data.f[i] = (float)(LIB_RNG_GetRandomNumber() % 1000) / 1000.0f;
LIB_SERIAL_Transmit(input->data.f, SIZE_INPUT, TYPE_F32);
#endif
LIB_MODEL_Run(&output);
LIB_SERIAL_Transmit(output->data.f, SIZE_OUTPUT, TYPE_F32);
HAL_Delay(1000);
}
}