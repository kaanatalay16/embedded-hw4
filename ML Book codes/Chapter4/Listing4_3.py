import py_serial
import numpy as np

py_serial.SERIAL_Init("COM4")

while 1:
    rqType, datalength, dataType = py_serial.SERIAL_PollForRequest()
    if rqType == py_serial.MCU_WRITES:
        data = py_serial.SERIAL_Read()
        
    elif rqType == py_serial.MCU_READS:
        sendArray = np.frombuffer(b'Hello World\n', dtype=np.uint8)
        py_serial.SERIAL_Write(sendArray)
        