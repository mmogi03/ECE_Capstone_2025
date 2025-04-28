#include <util/atomic.h>
#include <PinChangeInterrupt.h>
#include <ArduinoJson.h>


// Pin map
// Motor1  (pitch – “wood”)
#define ENC1A 3     // Yellow?
#define ENC1B 2     // INT1
#define PWM1  10
#define IN1A  7
#define IN1B  6
#define read1A bitRead(PIND, 2)
#define read1B bitRead(PIND, 3)

// Motor2  (yaw – “mirror”)
#define ENC2A 4     
#define ENC2B 5     
#define PWM2  11
#define IN2A  8
#define IN2B  9
#define read2A bitRead(PIND, 4)
#define read2B bitRead(PIND, 5)

//Globals
volatile long posi = 0;      // encoder count motor1
volatile long posj = 0;      // encoder count motor2

long   prevTPitch = 0, prevTYaw = 0;
float  ePrevPitch = 0,  ePrevYaw = 0;
float  iPitch = 0,      iYaw  = 0;

// controller constants
const float KP_PITCH = 2.75, KD_PITCH = 0.01, KI_PITCH = 0.2;
const float KP_YAW   = 1.5, KD_YAW   = 0.0025, KI_YAW   = 0.0;
const long  POS_TOL  = 8;     // "sensitivity"

// Initialize pos of encoders
long curPitch = 0, curYaw = 0;
static long targetPitch =  0;   
static long targetYaw   = 0;  

// Flag for reading 
bool flagRead = false;
String input2;



void setup() {
  Serial.begin(19200);
  //Serial.println("setup");
  // Motor1 encoder
  pinMode(ENC1A, INPUT);
  pinMode(ENC1B, INPUT);
  attachInterrupt(digitalPinToInterrupt(ENC1A), readEnc1A, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC1B), readEnc1B, CHANGE);

  // Motor1 driver
  pinMode(PWM1, OUTPUT);
  pinMode(IN1A, OUTPUT);
  pinMode(IN1B, OUTPUT);

  // Motor2 encoder
  pinMode(ENC2A, INPUT);
  pinMode(ENC2B, INPUT);
  attachPCINT(digitalPinToPCINT(ENC2A), readEnc2A, CHANGE);
  attachPCINT(digitalPinToPCINT(ENC2B), readEnc2B, CHANGE);

  // Motor2 driver
  pinMode(PWM2, OUTPUT);
  pinMode(IN2A, OUTPUT);
  pinMode(IN2B, OUTPUT);

  // Writing Serial Initial Values: 

  ATOMIC_BLOCK(ATOMIC_RESTORESTATE){
    while(!Serial){
      ;
    }
      ArduinoJson::JsonDocument doc;
      StaticJsonDocument<200> _doc;
      doc.set(_doc);            
      doc["pitch"] = 0;
      doc["yaw"] = 0;
      serializeJson(doc, Serial);
      Serial.println(); // Newline terminator.
      //Serial.println("setup");
    }
}


void loop() {

  // Read atomically for encoders
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { curPitch = posi; }
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { curYaw   = posj; }

  // Read atomically for serial communication: targets
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE){
    if (Serial.available()>0){
      input2 = Serial.readStringUntil('\n');
      }
  }
  if (input2.length() > 0){
    flagRead = true;
  }
  
  if (flagRead) {
    StaticJsonDocument<200> doc;
    DeserializationError error = deserializeJson(doc, input2);
    if (!error){
      // Retrieve the updated angle values using as<int>()
      targetPitch = doc["pitch"].as<int>();
      targetYaw = doc["yaw"].as<int>();
    }
  }

  // Call PIDs only when outside tolerance
  if (abs(targetPitch - curPitch) > POS_TOL) {
    PID_Pitch(targetPitch, curPitch);
  }
  if (abs(targetYaw - curYaw) > POS_TOL) {
    PID_Yaw(targetYaw, curYaw);
  }

  if (flagRead == true){
    ArduinoJson::JsonDocument response;
    StaticJsonDocument<200> _response;
    response.set(_response);   
    response["pitch"] = curPitch;
    response["yaw"] = curYaw;
    char output[128];
    serializeJson(response, output, sizeof(output));
    Serial.println(output);
    Serial.println();
    flagRead = false;
   }
   
  
  // Debug
  //Serial.print("P:"); Serial.print(curPitch);
  //Serial.println();
  //Serial.print(" / "); Serial.print(targetPitch);
  //Serial.print(" | Y:"); Serial.print(curYaw);
  //Serial.print(" / "); Serial.println(targetYaw);

  delay(500); 
}

/* =================================================================
 *                       PID FUNCTIONS
 * =================================================================*/
void PID_Pitch(long target, long current) {
  //Serial.println("PID Pitch");
  long now  = micros();
  float dt  = (now - prevTPitch) * 1e-6;   
  prevTPitch = now;

  float err   = -(float)target + current;
  float dErr  = (err - ePrevPitch) / dt;
  iPitch     += err * dt;

  float u = KP_PITCH * err + KD_PITCH * dErr + KI_PITCH * iPitch;


  // motor power
  float pwr = fabs(u);
  if( pwr > 255 ){
    pwr = 255;
  }

  // motor direction
  int dir = 1;
  if(u>0){
    dir = -1;
  }
  //Serial.println("value u");
  //Serial.println(pwr);
  // signal the motor
  setMotor(dir, pwr,  PWM1, IN1A, IN1B);
  delay(20);
  setMotor(0, 0,  PWM1, IN1A, IN1B);
  //Serial.println("blah");
  
  
  ePrevPitch = err;

}

void PID_Yaw(long target, long current) {
  //Serial.println("Going into PID yaw");
  long now  = micros();
  float dt  = (now - prevTYaw) * 1e-6;     
  prevTYaw = now;

  float err   = (float)target - current;
  float dErr  = (err - ePrevYaw) / dt;
  iYaw       += err * dt;

  float u = KP_YAW * err + KD_YAW * dErr + KI_YAW * iYaw;
  
  // motor power
  float pwr = fabs(u);
  if( pwr > 255 ){
    pwr = 255;
  }

  // motor direction
  int dir = 1;
  if(u<0){
    dir = -1;
  }

  // signal the motor
  setMotor(dir, u,  PWM2, IN2A, IN2B);
  delayMicroseconds(10000);
  setMotor(0, 0,  PWM2, IN2A, IN2B);
    
  ePrevYaw = err;
  //delay(10);
}

/* =================================================================
 *                     LOW‑LEVEL MOTOR HELPERS
 * =================================================================*/
void stepMotor1(int dir) {
  // Minimum usable pulse: 200 PWM
  setMotor(dir, 255, PWM1, IN1A, IN1B);
  delayMicroseconds(10000);
  setMotor(0,   0,  PWM1, IN1A, IN1B); 
}

void stepMotor2(int dir) {
  // Minimum pulse: 255 PWM for 40 µs, then hold at 30 PWM
  //Serial.println("sm 2");
  setMotor(dir, 255, PWM2, IN2A, IN2B);
  delayMicroseconds(10000);
  setMotor(1,  30, PWM2, IN2A, IN2B);     // holding torque
}

void setMotor(int dir, int pwmVal, int pwm, int in1, int in2) {
  analogWrite(pwm, pwmVal);
  if (dir > 0) {           // forward
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
  } else if (dir < 0) {    // reverse
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
  } else {                 // off
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
  }
}

/* =================================================================
 *                        ENCODERs
 * =================================================================*/
void readEnc1A() {
  bool A = read1A;
  bool B = read1B;
  if (A == B) posi++; else posi--;
}
void readEnc1B() {
  bool A = read1A;
  bool B = read1B;
  if (A != B) posi++; else posi--;
}

void readEnc2A() {
  bool A = read2A;
  bool B = read2B;
  if (A == B) posj++; else posj--;
}
void readEnc2B() {
  bool A = read2A;
  bool B = read2B;
  if (A != B) posj++; else posj--;
}
