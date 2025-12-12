#include "mbed.h"
#include "lib_uart.h"
#include "lib_serial.h"

const float txArray[2] = {3.14159265358979323846f, 2.7182818284590452354f};

int main(void){
    LIB_UART_Init();
    while (1){
        LIB_SERIAL_Transmit((void*)txArray, 2, TYPE_F32);
        wait_us(1000000);
    }
}
