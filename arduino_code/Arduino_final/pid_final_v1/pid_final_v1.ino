#include <util/atomic.h>
#include <PinChangeInterrupt.h>
#include <ArduinoJson.h>


// Pin map
// Motor1  (yaw – “wood”)
#define ENC1A 2    // Yellow
#define ENC1B 3     // white
#define PWM1  10
#define IN1A  7
#define IN1B  6
#define read1A bitRead(PIND, 2)
#define read1B bitRead(PIND, 3)

// Motor2  (pitch – “mirror”)
#define ENC2A 5     
#define ENC2B 4     
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
const float KP_PITCH = 0.9, KD_PITCH = 0.090, KI_PITCH = 0.002;
const float KP_YAW   = 0.8, KD_YAW   = 0.06, KI_YAW   = 0.065;
const long  POS_TOL  =  18;     // "sensitivity"

// Flags
bool flagRead = false;
bool flagPitch = false, flagYaw = false;
String input2;


// Initialize pos of encoders
long curPitch = 0, curYaw = 0;

static long targetPitch =  0;    // Looking from above: Positive = counterclockwise
static long targetYaw   =  0;  // Positive = mirror rotates up



// convergence timer
/*
bool   isMoving    = false;
bool atPitch = false;
bool atYaw = false;
unsigned long startTime = 0;
int counter = 0;
*/


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
      StaticJsonDocument<200> doc;       
      doc["pitch"] = 0;
      doc["yaw"] = 0;
      serializeJson(doc, Serial);
      Serial.println(); // Newline terminator.
      //Serial.println("setup");
    }
    delay(4000);
}


void loop() {

  
  // 1. read encoders
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { curPitch = posi; }
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { curYaw   = posj; }
//  Serial.print("Curr pitch: "); Serial.println(curPitch);
//  Serial.print("Curr yaw: "); Serial.println(curYaw);
  


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
      targetPitch = doc["yaw"].as<int>();
      targetYaw = doc["pitch"].as<int>();
      input2 = "";      
    }
    //Serial.println("received angles on arduino");
    if (targetPitch > 700) targetPitch = 700;
    else if (targetPitch < 20) targetPitch = 20;

    if (targetYaw > 100) targetYaw = 100;
    else if (targetYaw < -500 ) targetYaw = -500;
  }
  // 2. run PIDs if outside tolerance
  if (abs(targetPitch - curPitch) > POS_TOL) {
    PID_Pitch(targetPitch);
    flagPitch = false;
    //atPitch = false;
  }
  else{
    //atPitch = true;
    flagPitch = true;
  }
  
  if (abs(targetYaw   - curYaw  ) > POS_TOL) {
    PID_Yaw  (targetYaw);
    //atYaw = false;
    flagYaw = false;
  }
  else{
    //atYaw = true;
    flagYaw = true;
  }
// Writing Serial when 3 flags are true
  if (flagRead && flagPitch && flagYaw){
      ATOMIC_BLOCK(ATOMIC_RESTORESTATE){
    while(!Serial){
      ;
    }
      StaticJsonDocument<200> doc;          
      doc["pitch"] = curPitch;
      doc["yaw"] = curYaw;
      serializeJson(doc, Serial);
      Serial.println();
    }
    flagRead = false;
  }
  

    delay(500);
}

  
  /*
  if (!isMoving && (!atPitch || !atYaw)) {
    isMoving  = true;
    startTime = micros();
  }
  
  if (isMoving && atPitch && atYaw) {
    unsigned long elapsed = micros() - startTime;
    Serial.print("Converged in ");
    Serial.print(elapsed / 1000.0, 1);  // ms
    Serial.println(" ms");
    isMoving = false;
   
    if (counter%2 == 0){
      targetPitch = 150;
      targetYaw = 100;
    }
    else{
      targetPitch =  700;
      targetYaw   = -150; 
    }
    counter ++;
    delay(1500);
  }
    // 4. debug prints
  Serial.print("TargetPitch: "); Serial.println(targetPitch);
  Serial.print("  curPitch:  "); Serial.println(curPitch);
  Serial.println();
  Serial.print("TargetYaw:   "); Serial.println(targetYaw);
  Serial.print("  curYaw:    "); Serial.println(curYaw);
  Serial.println();
  delay(500);
  */




/* =================================================================
 *                       PID FUNCTIONS
 * =================================================================*/


void PID_Pitch(long target) {

  //Serial.print("at pitch");

  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { curPitch = posi; }

  float err = (float)target - curPitch;
  long now = micros();
  float dt = (now - prevTPitch) * 1e-6;
  if (dt <= 0) dt = 1e-3;         
  prevTPitch = now;

  float dErr = (err - ePrevPitch) / dt;
  iPitch    += err * dt;           

  float u = KP_PITCH * err + KD_PITCH * dErr + KI_PITCH * iPitch;
//  Serial.print("Current u: "); Serial.println(u);
  float pwr = fabs(u);
  if (pwr > 255) pwr = 255;
  else if (pwr < 100) pwr = 130;

  int dir = (u >= 0) ?  -1 : 1;
  
  setMotor(dir, (int)pwr, PWM1, IN1A, IN1B);
  delay(15);
  setMotor(0, 0, PWM1, IN1A, IN1B);

  ePrevPitch = err;
}

void PID_Yaw(long target) {

  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { 
    curYaw = posj;
    delay(5);
    }

  float err = (float)target - curYaw;

  long now = micros();
  float dt = (now - prevTYaw) * 1e-6;
  if (dt <= 0) dt = 1e-3;
  prevTYaw = now;

  float dErr = (err - ePrevYaw) / dt;
  iYaw     += err * dt;

  float u = KP_YAW * err + KD_YAW * dErr + KI_YAW * iYaw;

  float pwr = fabs(u);
  if (pwr > 255) pwr = 255;
  else if(pwr<80){
    pwr = 80;
  }

  int dir = (u >= 0) ?  -1 : 1;

  if (dir > 0){
    setMotor(dir, (int)pwr, PWM2, IN2A, IN2B);
    delay(5);
    setMotor(0, 0, PWM2, IN2A, IN2B);
  }
  else{
    setMotor(dir, (int)pwr, PWM2, IN2A, IN2B);
    delay(45);
    setMotor(0, 0, PWM2, IN2A, IN2B);   
  }


  ePrevYaw = err;
}



/* =================================================================
 *                     LOW‑LEVEL MOTOR HELPERS
 * =================================================================*/

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
