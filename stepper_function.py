import time
import threading
import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib

DIR1 = 21
STEP1 = 20
GPIO_PINS = (0, 0, 0)

DIR2 = 15
STEP2 = 14

stepper1 = RpiMotorLib.A4988Nema(DIR1, STEP1, GPIO_PINS, "DRV8825")
stepper2 = RpiMotorLib.A4988Nema(DIR2, STEP2, GPIO_PINS, "DRV8825")

STEP_DELAY = 0.0005

def mode_plastic_1():

    def plastic_1(): 
        stepper1.motor_go(False, "Full", 800, STEP_DELAY, False, 0.05)

    def plastic_2():
        stepper2.motor_go(False, "Full", 400, STEP_DELAY, False, 0.05)

    t1 = threading.Thread(target=plastic_1)
    t2 = threading.Thread(target=plastic_2)

    t1.start()
    time.sleep(0.5)
    t2.start()

    t1.join()
    t2.join()

def mode_plastic_2():

    def plastic_1(): 
        stepper1.motor_go(True, "Full", 800, STEP_DELAY, False, 0.05)

    def plastic_2():
        stepper2.motor_go(True, "Full", 400, STEP_DELAY, False, 0.05)

    t1 = threading.Thread(target=plastic_1)
    t2 = threading.Thread(target=plastic_2)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

def mode_can_1():
    def can_1(): 
        stepper1.motor_go(True, "Full", 400, STEP_DELAY, False, 0.05)

    def can_2():
        stepper2.motor_go(False, "Full", 400, STEP_DELAY, False, 0.05)

    t1 = threading.Thread(target=can_1)
    t2 = threading.Thread(target=can_2)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

def mode_can_2():
    def can_1(): 
        stepper1.motor_go(False, "Full", 400, STEP_DELAY, False, 0.05)

    def can_2():
        stepper2.motor_go(True, "Full", 400, STEP_DELAY, False, 0.05)

    t1 = threading.Thread(target=can_1)
    t2 = threading.Thread(target=can_2)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

def mode_paper_1():
    def paper_1(): 
        stepper1.motor_go(True, "Full", 400, STEP_DELAY, False, 0.05)

    def paper_2():
        stepper2.motor_go(True, "Full", 800, STEP_DELAY, False, 0.05)

    t1 = threading.Thread(target=paper_1)
    t2 = threading.Thread(target=paper_2)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

def mode_paper_2():
    def paper_1(): 
        stepper1.motor_go(False, "Full", 400, STEP_DELAY, False, 0.05)

    def paper_2():
        stepper2.motor_go(False, "Full", 800, STEP_DELAY, False, 0.05)

    t1 = threading.Thread(target=paper_1)
    t2 = threading.Thread(target=paper_2)

    t1.start()
    t2.start()

    t1.join()
    t2.join()


def execute_command(mode):

    if mode == 1:
        mode_plastic_1()
        time.sleep(1)
        mode_plastic_2()
    elif mode == 2:
        mode_can_1()
        time.sleep(1)
        mode_can_2()
    elif mode == 3:
        mode_paper_1()
        time.sleep(1)
        mode_paper_2()

try:
    execute_command(3)

finally:
    GPIO.cleanup