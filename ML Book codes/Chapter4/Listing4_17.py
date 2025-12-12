import py_serialimg
import numpy as np

py_serialimg.SERIAL_Init("COM4")

mandrill = "mandrill.tif"
while 1:
    rqType, height, width, format  = py_serialimg.SERIAL_IMG_PollForRequest()
    if rqType == py_serialimg.MCU_WRITES:
        img = py_serialimg.SERIAL_IMG_Read()
    elif rqType == py_serialimg.MCU_READS:
        img = py_serialimg.SERIAL_IMG_Write(mandrill)


