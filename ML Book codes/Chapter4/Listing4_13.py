import py_serial
import numpy as np
from scipy.io.wavfile import write

py_serial.SERIAL_Init("COM4")

sampleRate = 16000
while 1:
    rqType, datalength, dataType = py_serial.SERIAL_PollForRequest()
    if rqType == 87:
        data = py_serial.SERIAL_Read()
        channel1 = np.array(data[0::2], dtype=np.int16)
        channel2 = np.array(data[1::2], dtype=np.int16)
        data = np.transpose(np.array([channel1,channel2], dtype=np.int16))
        print(np.shape(data))
        write('test.wav', sampleRate, data.astype(np.int16))