import ctypes
import numpy as np
import time

class api_teleoperate_sm:
    def __init__(self):
        self.absolute_lib_path = R"C:\Users\ktang\Desktop\Kai\Controller_Ver.5.0.6\Themes\Kai\imitation_learning\intime_dll\x64\Release\Sewing_DLL.dll"
        self.api_teleoperate = ctypes.WinDLL(self.absolute_lib_path, winmode=0x8)

        # define the function provided by the dll
        # robot A
        self.Init_SM_A_Teleop = self.api_teleoperate.Py_Init_SM_Teleoperate_A # initialize the robot A
        self.Init_SM_A_Teleop.restype = ctypes.c_int

        # robot B
        self.Init_SM_B_Teleop = self.api_teleoperate.Py_Init_SM_Teleoperate_B # initialize the robot B
        self.Init_SM_B_Teleop.restype = ctypes.c_int

        # teleoperate related
        self.Init_SM_Teleop = self.api_teleoperate.Py_Init_SM_Teleoperation # initialize teleoperate
        self.Init_SM_Teleop.restype = ctypes.c_int

        self.teleoperate_w2i = self.api_teleoperate.Py_Teleoperation_w2i # initialize shared memory for control sewing
        self.teleoperate_w2i.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double))

        # initialization
        self.init_check = self.Init_SM_A_Teleop()
        if self.init_check == 0:
            print("Robot A Initialization successful")
        else:
            print("Robot A Initialization failed")
        
        self.init_check = self.Init_SM_B_Teleop()
        if self.init_check == 0:
            print("Robot B Initialization successful")
        else:
            print("Robot B Initialization failed")
        
        self.init_check = self.Init_SM_Teleop()

    def write_desFT(self, des_FT_A, des_FT_B):
        desFT_A_np = np.array(des_FT_A, dtype=np.float64)
        desFT_B_np = np.array(des_FT_B, dtype=np.float64)
        des_FT_A_double = desFT_A_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        des_FT_B_double = desFT_B_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        # Call the teleoperation function with the ctypes pointers
        self.teleoperate_w2i(des_FT_A_double, des_FT_B_double)
        return
    
if __name__ == "__main__":
    api_teleoperate = api_teleoperate_sm()
    for i in range(10):
        des_FT_A = [0.1+i, 0.2+i, 0.3+i, 0.4+i, 0.5+i, 0.6+i]
        des_FT_B = [0.7+i, 0.8+i, 0.9+i, 1.0+i, 1.1+i, 1.2+i]
        api_teleoperate.write_sewing_feedback(des_FT_A, des_FT_B)
        time.sleep(1)
    