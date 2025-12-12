#include "mbed.h"
#include "lib_uart.h"
#include "lib_serial.h"
#include <string.h>

uint8_t rxArray[12] = {0};
DigitalOut led1(LED1);

int main(void){
    LIB_UART_Init();

    while (1){
        LIB_SERIAL_Receive(rxArray, 12, TYPE_U8);
        if (std::strncmp((const char*)rxArray, "Hello World\n", 12) == 0){
            led1 = !led1;
        }
        wait_us(1000000);
    }
}