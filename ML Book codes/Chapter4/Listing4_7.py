
import py_serial
import numpy as np

py_serial.SERIAL_Init("COM4")

while 1:
    rqType, datalength, dataType = py_serial.SERIAL_PollForRequest()
    if rqType == 87:
        data = py_serial.SERIAL_Read()
        print("Humidity: " + str(data[0]))
        print("Temperature: " + str(data[1]))