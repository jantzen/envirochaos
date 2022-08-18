import piplates.DAQCplate as DAQC
import RPi.GPIO as GPIO
from Adafruit_BME280 import *
import time
import pickle
import socket
import os
import numpy as np

# create a variable (that will be used globally) to hold data
data = []

def save(channel):
    """ Callback function for button interrupt that causes the current  
    timseries to be saved.
    """
    time.sleep(0.1)
    
    print("Button pushed!")
            
    DAQC.getINTflags(0) # clear the interrupt
    
#    # use light strobes to indicate measurement
#    DAQC.setDOUTall(0,0)
#    for j in range (7):
#                    DAQC.setDOUTbit(0,j)
#                    time.sleep(0.1)
#                    DAQC.clrDOUTbit(0,j)
#                    time.sleep(0.1)
    
#    # measure from the BME280 (ORDER MATTERS!)
#    temp = sensor.read_temperature()
#    pressure = sensor.read_pressure() / 100. # convert to hectopascals
#    humidity = sensor.read_humidity()
#    t = sensor.t_fine
#    
#    # measure from the (analog) photo-sensor
#    light = DAQC.getADC(0,0)

    # set filename
    archive_time = time.ctime()
    archive_time = archive_time.replace(' ', '_')
    filename = ('../../data/output/lightbulb_data_' + archive_time + '.txt')
   
    global data
    np.savetxt(filename, np.array(data))
#    data.append([t, temp, pressure, humidity, light])
    
    # turn off LEDs
    DAQC.setDOUTall(0,0)
        

# set up GPIO interface for pushbotton on PiDAQC
DAQC.enableSWint(0) # enable pushbutton interrupt
DAQC.intEnable(0) # enable global interrupts
DAQC.getINTflags(0) # clear any flags

# GPIO event detection must be set up AFTER clearing DAQC interrupt flag
GPIO.setmode(GPIO.BCM)
GPIO.setup(22, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.add_event_detect(22, GPIO.FALLING, callback=sample)
DAQC.getINTflags(0)

# set up BME280
sensor = BME280(mode=BME280_OSAMPLE_8)


try:
    DAQC.setLED(0,0)
    while(1):
        # measure from the (analog) photo-sensor
        light = DAQC.getADC(0,0)
        data.append(light)
        print("{} \r".format(light), end='')

        # wait 
        time.sleep(0.1)

except KeyboardInterrupt:
    DAQC.clrLED(0,0)
    GPIO.cleanup()
    
    # set filename
    archive_time = time.ctime()
    archive_time = archive_time.replace(' ', '_')
    filename = ('../../data/output/lightbulb_data_' + archive_time + '.txt')
    np.savetxt(filename, np.array(data))
