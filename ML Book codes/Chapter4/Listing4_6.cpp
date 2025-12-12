#include "mbed.h"
#include "lib_hts221.h"
#include "lib_serial.h"
#include "lib_uart.h"
#include "lib_i2c.h"

float hts221[2] = {0};
int main()
{
    SCB_EnableDCache();
    SCB_EnableICache();
    LIB_I2C1_Init();
    LIB_UART_Init();
    LIB_HTS221_Init();
    while (true)
    {
        LIB_HTS221_GetHumidity(&hts221[0]);
        LIB_HTS221_GetTemperature(&hts221[1]);
        LIB_SERIAL_Transmit(hts221, 2, TYPE_F32);
        HAL_Delay(1000);
    }
}
