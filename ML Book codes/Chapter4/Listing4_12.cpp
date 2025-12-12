#include "mbed.h"
#include "lib_serial.h"
#include "lib_uart.h"
#include "lib_audio.h"

#define BUFFER_SIZE (32000 * 3)

uint16_t AudioBuffer[BUFFER_SIZE] = {0};

int main(){
    SCB_EnableICache();
    SCB_EnableDCache();
    LIB_UART_Init();
    LIB_AUDIO_Init();
    
    while (true){
        LIB_AUDIO_StartRecording(AudioBuffer, BUFFER_SIZE);
        if (LIB_AUDIO_PollForRecording(5000) == 0){
            LIB_SERIAL_Transmit(AudioBuffer, BUFFER_SIZE, TYPE_S16);
        }
    }
}