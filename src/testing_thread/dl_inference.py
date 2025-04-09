import time

def run_inference(auto_adjust_flag):
    while True:
        if auto_adjust_flag.value:
            print("DL Inference Running...")
            # Simulate DL processing delay.
            time.sleep(2)
            print("DL Inference: Angle sent to motors (simulated)")
        else:
            print("DL Inference Paused...")
            time.sleep(1)

if __name__ == '__main__':
    # For standalone testing.
    run_inference(auto_adjust_flag=type("Dummy", (), {"value": True})())
