#include "mbed.h"

DigitalOut led1(LED1);
Ticker ticker;

void TICKER_Callback(){
    led1 = !led1;
}

int main(){
    ticker.attach(&TICKER_Callback, 1);
    while (true){
        wait_us(1000);
    }
}
