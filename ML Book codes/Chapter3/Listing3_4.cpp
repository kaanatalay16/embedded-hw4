#include "mbed.h"

DigitalOut led1(LED1);
InterruptIn button(BUTTON1);

void BUTTON_Callback(){
    led1 = !led1;
}

int main(){
    button.rise(&BUTTON_Callback);
    
    while (true){
        wait_us(1000);
    }
}