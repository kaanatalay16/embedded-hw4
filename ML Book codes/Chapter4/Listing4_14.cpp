#include "mbed.h"
#include "BSP/STM32746G-Discovery/stm32746g_discovery_sdram.h"
#include "lib_mpu.h"
#include "lib_uart.h"
#include "lib_ov5640.h"
#include "lib_image.h"
#include "lib_serialimage.h"

const uint8_t * imageBuffer = (uint8_t *)0xC0000000;
IMAGE_HandleTypeDef img;

int main(){
    SCB_EnableICache();
    SCB_EnableDCache();
    LIB_UART_Init();
    LIB_MPU_Init();
    BSP_SDRAM_Init();
    LIB_IMAGE_InitStruct(&img, (uint8_t*)imageBuffer, IMAGE_RESOLUTION_VGA_HEIGHT, IMAGE_RESOLUTION_VGA_WIDTH, IMAGE_FORMAT_RGB565);
    LIB_OV5640_Init(OV5640_RESOLUTION_R640x480, OV5640_FORMAT_RGB565);
    
    while (true){
        if (!LIB_OV5640_CaptureSnapshot(&img, 5000)){
            LIB_SERIAL_IMG_Transmit(&img);
        }
    }
}