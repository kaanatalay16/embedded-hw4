import py_serial

py_serial.SERIAL_Init("COM4")

while 1:
    rqType, datalength, dataType = py_serial.SERIAL_PollForRequest()
    if rqType == 87:
        data = py_serial.SERIAL_Read()
        print("Acc X: " + str(data[0]) + " Acc Y: " + str(data[1]) + " Acc Z: " + str(data[2]))
        print("Gyro X: " + str(data[3]) + " Gyro Y: " + str(data[4]) + " Gyro Z: " + str(data[5]))
        print("Mag X: " + str(data[6]) + " Mag Y: " + str(data[7]) + " Mag Z: " + str(data[8]))
