#include "mbed.h"

mbed_stats_heap_t heapInfo;
mbed_stats_stack_t stackInfo;
uint32_t i;
uint32_t data[1000];

Timer myTimer;

int main()
{
myTimer.start();
for (i = 0; i < 1000; i++)
{
data[i] = i;
}
myTimer.stop();
printf("Execution time in microseconds: %llu\n", myTimer.elapsed_time());

mbed_stats_heap_get(&heapInfo);
printf("Heap size: %ld\n", heapInfo.reserved_size);
printf("Used heap: %ld\n", heapInfo.current_size);

mbed_stats_stack_get(&stackInfo);
printf("Main stack size: %ld\n", stackInfo.reserved_size);
printf("Used main stack: %ld\n", stackInfo.max_size);

while (true);
}