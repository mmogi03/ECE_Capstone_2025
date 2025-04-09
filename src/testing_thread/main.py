from multiprocessing import Process, Value
from ctypes import c_bool
import server
import dl_inference

if __name__ == '__main__':
    # Initialize the shared auto_adjust_flag to True so the DL inference starts running.
    auto_adjust_flag = Value(c_bool, True)
    
    # Create processes for the server and the DL inference script.
    p_server = Process(target=server.run_server, args=(auto_adjust_flag,))
    p_dl = Process(target=dl_inference.run_inference, args=(auto_adjust_flag,))
    
    # Start both processes.
    p_server.start()
    p_dl.start()
    
    # Wait for both processes to finish.
    p_server.join()
    p_dl.join()
