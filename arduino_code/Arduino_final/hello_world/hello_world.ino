#include <ArduinoJson.h>

void setup() {
  Serial.begin(9600);
  while (!Serial) {
    ; // Wait for serial port to connect (needed for native USB)
  }
  
  // Send base angle once on startup.
  StaticJsonDocument<200> doc;
  doc["pitch"] = 5;
  doc["yaw"] = 5;
  serializeJson(doc, Serial);
  Serial.println(); // Newline terminator.
}

void loop() {
  if (Serial.available() > 0) {
    // Read incoming command until newline.
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0) {
//      Serial.print("Received: ");
//      Serial.println(input);
      
      // Parse the incoming JSON.
      StaticJsonDocument<200> doc;
      DeserializationError error = deserializeJson(doc, input);
      if (!error) {
        // Retrieve the updated angle values using as<int>()
        int pitch = doc["pitch"].as<int>();  
        int yaw = doc["yaw"].as<int>();      
        
        // Simulate motor movement delay.
        delay(1000);
        
        // Print updated angles after simulated motor movement.
//        Serial.print("Motor moved to angles - Pitch: ");
//        Serial.print(pitch);
//        Serial.print(", Yaw: ");
//        Serial.println(yaw);
        
        // Prepare response with encoder values.
        // For simulation, increment the received values by 1.
        // Prepare response with encoder values.
        StaticJsonDocument<200> response;
        response["pitch"] = pitch + 1;
        response["yaw"] = yaw + 1;
        
        // Create a buffer and serialize the JSON response into it.
        char output[128];
        serializeJson(response, output, sizeof(output));
        
        // Print the serialized response.
//        Serial.print("Sending encoder values: ");
        Serial.println(output);
        
        // Send the response over Serial (if needed, you can also send the buffer directly).
//        Serial.println(); // Newline terminator.
      } else {
        Serial.println("Invalid JSON received");
      }
    }
  }
}
