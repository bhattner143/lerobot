import ctypes
import numpy as np
import time

class api_sewing_sm:
    def __init__(self):
        
        class States(ctypes.Structure):
            _fields_ = [("stateA", ctypes.c_int),
                        ("stateB", ctypes.c_int)]

        self.absolute_lib_path = R"C:\Users\ktang\Desktop\Kai\Controller_Ver.5.0.6\Themes\Kai\imitation_learning\intime_dll\x64\Release\Sewing_DLL.dll"
        self.api_sewing_sm = ctypes.WinDLL(self.absolute_lib_path, winmode=0x8)

        # define the function provided by the dll
        # robot A
        self.Init_SM_A_Sewing = self.api_sewing_sm.Py_Init_SM_A_Sewing # initialize the robot A
        self.Init_SM_A_Sewing.restype = ctypes.c_int

        # robot B
        self.Init_SM_B_Sewing = self.api_sewing_sm.Py_Init_SM_B_Sewing # initialize the robot B
        self.Init_SM_B_Sewing.restype = ctypes.c_int

        # sewing related
        self.Init_SM_SewingFeedback = self.api_sewing_sm.Py_Init_SM_SewingFeedback # initialize shared memory for control sewing
        self.Init_SM_SewingFeedback.restype = ctypes.c_int

        self.read_sewing_speed_dual = self.api_sewing_sm.Py_Read_Sewing_Speed
        self.read_sewing_speed_dual.restype = ctypes.c_double

        self.read_sewing_state = self.api_sewing_sm.Py_Read_Sewing_State
        self.read_sewing_state.restype = States

        self.read_sewn_stitch = self.api_sewing_sm.Py_Read_Final_Sewn_Stitches
        self.read_sewn_stitch.restype = ctypes.c_int

        self.write_sewing_feedback_dual = self.api_sewing_sm.Py_Write_Sewing_Feedback
        # four input, double, int, bool, unsigned int
        self.write_sewing_feedback_dual.argtypes = (ctypes.c_double, ctypes.c_int, ctypes.c_bool, ctypes.c_uint)
        self.write_sewing_feedback_dual.restype = ctypes.c_int
        

        self.init_check = self.Init_SM_A_Sewing()
        if self.init_check == 0:
            print("Robot A Initialization successful")
        else:
            print("Robot A Initialization failed")
        self.init_check = self.Init_SM_B_Sewing()
        if self.init_check == 0:
            print("Robot B Initialization successful")
        else:
            print("Robot B Initialization failed")
        self.init_check = self.Init_SM_SewingFeedback()

    def write_sewing_feedback(self, rotation_speed, desired_stitch, sewFlag, finFlag = False):
        check_flag = self.write_sewing_feedback_dual(rotation_speed, desired_stitch, finFlag, sewFlag)
        # if check_flag == 1:
        #     print(f'write rotation speed : {rotation_speed}')
        return
    
    def read_sewing_speed(self):
        sewing_speed = self.read_sewing_speed_dual()
        return sewing_speed
    
    def read_sewn_stitch_num(self):
        stitch_num = self.read_sewn_stitch()
        return stitch_num
    
if __name__ == "__main__":
    api_sewing = api_sewing_sm()
    sewing_speed = api_sewing.read_sewing_speed()
    state = api_sewing.read_sewing_state()
    print(sewing_speed)
    print(state.stateA)
    print(state.stateB)
    sewn_stitch_num = api_sewing.read_sewn_stitch_num()
    print(sewn_stitch_num)

    