import ctypes
import numpy as np
import time

class api_positioning_sm:
    def __init__(self):

        class States(ctypes.Structure):
            _fields_ = [("stateA", ctypes.c_int),
                        ("stateB", ctypes.c_int),
                        ("state_vs", ctypes.c_int)]

        self.absolute_lib_path = R"C:\Users\ktang\Desktop\Kai\Controller_Ver.5.0.6\Themes\Kai\imitation_learning\intime_dll\x64\Release\Sewing_DLL.dll"
        self.api_positioning_sm = ctypes.WinDLL(self.absolute_lib_path, winmode=0x8)

        # define the function provided by the dll
        # robot A
        self.Init_SM_A_Positioning = self.api_positioning_sm.Py_Init_SM_A_Positioning # initialize the robot A
        self.Init_SM_A_Positioning.restype = ctypes.c_int

        # robot B
        self.Init_SM_B_Positioning = self.api_positioning_sm.Py_Init_SM_B_Positioning # initialize the robot B
        self.Init_SM_B_Positioning.restype = ctypes.c_int

        # loading and alignment
        self.read_loading_alignment = self.api_positioning_sm.Py_Read_Loading_Alignment_State
        self.read_loading_alignment.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double))
        self.read_loading_alignment.restype = States

        self.write_loading_alignment_dual = self.api_positioning_sm.Py_Write_Loading_Alignment
        self.write_loading_alignment_dual.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_bool)
        self.write_loading_alignment_dual.restype = ctypes.c_int

        self.init_check = self.Init_SM_A_Positioning()
        if self.init_check == 0:
            print("Robot A Initialization successful")
        else:
            print("Robot A Initialization failed")
        self.init_check = self.Init_SM_B_Positioning()
        if self.init_check == 0:
            print("Robot B Initialization successful")
        else:
            print("Robot B Initialization failed")

    def read_loading_alignment_state(self):
        curEndPos_A = np.zeros((6), np.float64)
        curEndPos_B = np.zeros((6), np.float64)
        States = self.read_loading_alignment(curEndPos_A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                          curEndPos_B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return curEndPos_A, curEndPos_B, States

    def write_loading_alignment_feedback(self, v_x_A, v_y_A, omega_z_A, v_x_B, v_y_B, omega_z_B, finFlag = False):
        check_flag = self.write_loading_alignment_dual(v_x_A, v_y_A, omega_z_A, v_x_B, v_y_B, omega_z_B, finFlag)
        return
    
if __name__ == "__main__":
    api_positioning = api_positioning_sm()
    curEndPos_A, curEndPos_B, States = api_positioning.read_loading_alignment_state()
    print(curEndPos_A)
    print(curEndPos_B)
    print(States.stateA)
    print(States.stateB)
    print(States.state_vs)

