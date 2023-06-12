import numpy as np
from math import pi
import math
import time

def grasped(graspclient):
    slave = 65
    flag = graspclient.read_input_registers(268,1,unit=slave).registers[0]
    flag = (flag&0x02) == 2

    if flag:
        print("Grasp detected: True")
    return flag

def Graspable(graspclient):
    slave = 65
    flag = graspclient.read_input_registers(268,1,unit=slave).registers[0]
    flag = (flag&0x08) == 8

    if flag:
        print("Grasp availablity: False")
    
    return flag

def resetTool(graspclient):
    print('Tool reseting')
    toolslave = 63
    graspclient.write_register(0,2,unit=toolslave)
    time.sleep(3)
    print("Reset Fininshed", end='\r')

def closeGrasp(force,width,graspclient):
    # If grasped, reset&openGrasp
    if grasped(graspclient):
        resetTool(graspclient)
        openGrasp(400,1000,graspclient)
    # If S1 activated, reset
    if Graspable(graspclient):
        resetTool(graspclient)
    slave = 65
    graspclient.write_registers(0,[force,width,1],unit=slave)
    time.sleep(1)

def openGrasp(force,width,graspclient):
    # If S1 activated, reset
    FLAG = True 
    while FLAG:
        FAIL = Graspable(graspclient)
        if FAIL:
            resetTool(graspclient)
        else: 
            FLAG=False
    # if Graspable(graspclient):
    #     resetTool(graspclient)
    slave = 65
    graspclient.write_registers(0,[force,width,1],unit=slave)
    time.sleep(1)

