#include "mbed.h"

#define WAIT_TIME_MS 500
DigitalOut greenLED(LED1);

int main()
{
printf("This is the bare metal blinky example running on Mbed OS %d.%d.%d.\n", MBED_MAJOR_VERSION, MBED_MINOR_VERSION, MBED_PATCH_VERSION);

while (true)
{
greenLED = !greenLED;
thread_sleep_for(WAIT_TIME_MS);
}
}