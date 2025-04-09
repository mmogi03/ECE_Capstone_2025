import RPi.GPIO as GPIO
import time

# Define pins (update with correct Raspberry Pi GPIO numbers)
# Original C++ defines: ENCA = 2, ENCB = 3, PWM = 5, IN2 = 6, IN1 = 7.
# Here we use example BCM numbers. Change them to your actual wiring.
ENCA = 24    # Example: Encoder channel A (Yellow)
ENCB = 23    # Example: Encoder channel B (White)
PWM_PIN = 12 # Example: PWM pin (must be PWM-capable)
IN1 = 17     # Example: Motor control input 1
IN2 = 27     # Example: Motor control input 2

# Global variable for encoder position and state tracking for quadrature decoding
pos = 0
# We'll combine the two encoder signals into a 2-bit number: (ENCA << 1) | ENCB.
last_state = 0

def encoder_callback(channel):
    """
    This callback is triggered by changes on either encoder channel.
    It reads both channels and uses state transitions to update the encoder position.
    """
    global pos, last_state
    # Read current states of both encoder channels
    A = GPIO.input(ENCA)
    B = GPIO.input(ENCB)
    current_state = (A << 1) | B

    # Only process if state has changed (this avoids duplicate processing)
    if current_state != last_state:
        # Define the valid state transitions for forward rotation:
        # 00 -> 01, 01 -> 11, 11 -> 10, 10 -> 00
        forward = [(0, 1), (1, 3), (3, 2), (2, 0)]
        # And for reverse rotation:
        # 00 -> 10, 10 -> 11, 11 -> 01, 01 -> 00
        reverse = [(0, 2), (2, 3), (3, 1), (1, 0)]
        
        transition = (last_state, current_state)
        if transition in forward:
            pos += 1
        elif transition in reverse:
            pos -= 1
        # Update the last state for the next transition
        last_state = current_state

def set_motor(direction, pwm_val, pwm_instance, in1_pin, in2_pin):
    """
    Controls the motor's direction and speed.
    
    Args:
        direction (int): 1 for forward, -1 for reverse, 0 for stop.
        pwm_val (int or float): PWM duty cycle (0-100).
        pwm_instance: The PWM instance controlling speed.
        in1_pin (int): GPIO pin for motor control input 1.
        in2_pin (int): GPIO pin for motor control input 2.
    """
    # Set PWM duty cycle (speed)
    pwm_instance.ChangeDutyCycle(pwm_val)
    # Set motor direction based on the input
    if direction == 1:
        GPIO.output(in1_pin, GPIO.HIGH)
        GPIO.output(in2_pin, GPIO.LOW)
    elif direction == -1:
        GPIO.output(in1_pin, GPIO.LOW)
        GPIO.output(in2_pin, GPIO.HIGH)
    else:
        GPIO.output(in1_pin, GPIO.LOW)
        GPIO.output(in2_pin, GPIO.LOW)

def setup():
    """
    Initializes the GPIO pins, serial (via print statements), and interrupts.
    This mirrors the setup() function in the C++ code.
    """
    global last_state
    # Use Broadcom SOC channel numbering
    GPIO.setmode(GPIO.BCM)
    
    # Setup encoder pins with pull-up resistors
    GPIO.setup(ENCA, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(ENCB, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    
    # Setup motor control pins
    GPIO.setup(PWM_PIN, GPIO.OUT)
    GPIO.setup(IN1, GPIO.OUT)
    GPIO.setup(IN2, GPIO.OUT)
    
    # Setup PWM on the PWM_PIN at 1kHz frequency and start at 0% duty cycle
    global pwm
    pwm = GPIO.PWM(PWM_PIN, 1000)
    pwm.start(0)
    
    # Initialize the last state from current encoder readings
    last_state = (GPIO.input(ENCA) << 1) | GPIO.input(ENCB)
    
    # Attach interrupts on both encoder channels on both rising and falling edges
    GPIO.add_event_detect(ENCA, GPIO.BOTH, callback=encoder_callback)
    GPIO.add_event_detect(ENCB, GPIO.BOTH, callback=encoder_callback)
    
    print("Setup complete, interrupts attached.")

def loop():
    """
    Main loop that mimics the C++ loop() function.
    It alternates the motor direction and prints the encoder position.
    """
    global pos
    while True:
        print("\nNew iteration")
        # Set motor forward: direction 1, full speed (100% duty cycle)
        set_motor(1, 100, pwm, IN1, IN2)
        time.sleep(0.5)
        print("Encoder position:", pos)
        time.sleep(0.5)
        
        print("Other direction")
        # Set motor reverse: direction -1, lower speed (15% duty cycle)
        set_motor(-1, 15, pwm, IN1, IN2)
        time.sleep(5)
        print("Encoder position:", pos)

if __name__ == "__main__":
    try:
        setup()
        loop()
    except KeyboardInterrupt:
        print("\nExiting: stopping motor and cleaning up GPIO...")
    finally:
        pwm.stop()
        GPIO.cleanup()
