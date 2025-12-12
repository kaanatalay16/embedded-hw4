#include "mbed.h"
#include "lib_bno055.h"
#include "lib_serial.h"
#include "lib_uart.h"
#include "lib_i2c.h"

BNO055_F32DataTypeDef bno055Data;

int main()
{
    SCB_EnableICache();
    SCB_EnableDCache();
    LIB_I2C1_Init();
    LIB_UART_Init();
    LIB_BNO055_Init();

    while (true)
    {
        LIB_BNO055_ReadAccelXYZ(&bno055Data.accel[0], &bno055Data.accel[1], &bno055Data.accel[2]);
	    LIB_BNO055_ReadGyroXYZ(&bno055Data.gyro[0], &bno055Data.gyro[1], &bno055Data.gyro[2]);
	    LIB_BNO055_ReadMagXYZ(&bno055Data.mag[0], &bno055Data.mag[1], &bno055Data.mag[2]);
	    LIB_SERIAL_Transmit(&bno055Data, 9, TYPE_F32);
	    HAL_Delay(1000);
    }
}
