import ctypes
import numpy as np
import time

class api_robot:
    def __init__(self):

        self.absolute_lib_path = R"C:\Users\ktang\Desktop\Kai\Controller_Ver.5.0.6\Themes\Kai\imitation_learning\intime_dll\x64\Release\Sewing_DLL.dll"
        self.api_robot = ctypes.WinDLL(self.absolute_lib_path, winmode=0x8)

        # define the function provided by the dll
        # robot A
        self.Init_SM_A = self.api_robot.Py_Init_SM_A # initialize the robot A
        self.Init_SM_A.restype = ctypes.c_int

        self.read_angle_A = self.api_robot.Py_Read_SM_A # read the angle of robot A
        self.read_angle_A.argtype = ctypes.POINTER(ctypes.c_double)

        self.read_state_A = self.api_robot.Py_Read_Status_A # read the task state of robot A
        self.read_state_A.restype = ctypes.c_int

        # robot B
        self.Init_SM_B = self.api_robot.Py_Init_SM_B # initialize the robot B
        self.Init_SM_B.restype = ctypes.c_int

        self.read_angle_B = self.api_robot.Py_Read_SM_B # read the angle of robot B
        self.read_angle_B.argtype = ctypes.POINTER(ctypes.c_double)

        self.read_state_B = self.api_robot.Py_Read_Status_B # read the task state of robot B
        self.read_state_B.restype = ctypes.c_int

        # general master
        self.init_general_master = self.api_robot.Py_Init_SM_General_Master
        self.init_general_master.restype = ctypes.c_int

        self.write_task_state = self.api_robot.Py_Write_SM_General_Master
        self.write_task_state.argtypes = (ctypes.c_int,)

        self.read_task_init = self.api_robot.Py_Read_General_Task_Init_Flag
        self.read_task_init.restype = ctypes.c_int

        self.read_task_fin_A = self.api_robot.Py_Read_General_Task_End_Flag_A
        self.read_task_fin_A.restype = ctypes.c_int

        self.read_task_fin_B = self.api_robot.Py_Read_General_Task_End_Flag_B
        self.read_task_fin_B.restype = ctypes.c_int

        # sewing machine master
        self.init_sewing_machine_master = self.api_robot.Py_Init_SM_Sewing_Machine_Master
        self.init_sewing_machine_master.restype = ctypes.c_int

        self.sewing_machine_master = self.api_robot.Py_Write_SM_Sewing_Machine_Master 
        self.sewing_machine_master.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double))  # input is one int and one double array [10]

        # grasp and release master
        self.init_grasp_release_master = self.api_robot.Py_Init_SM_Grasp_Release_Master
        self.init_grasp_release_master.restype = ctypes.c_int

        self.grasp_release_master = self.api_robot.Py_Write_SM_Grasp_Release_Master
        self.grasp_release_master.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p) # input is two int and two char [1024]

        # end-effector movement master
        self.init_endeffector_movement_master = self.api_robot.Py_Init_SM_EndEffector_Movement_Master
        self.init_endeffector_movement_master.restype = ctypes.c_int

        self.endeffector_movement_master = self.api_robot.Py_Write_SM_EndEffector_Movement_Master
        self.endeffector_movement_master.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p) # input is two int and two char [1024]

        # fabric movement master
        self.init_fabric_movement_master = self.api_robot.Py_Init_SM_Fabric_Movement_Master
        self.init_fabric_movement_master.restype = ctypes.c_int

        self.fabric_movement_master = self.api_robot.Py_Write_SM_Fabric_Movement_Master
        self.fabric_movement_master.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p)

        # sewing master
        self.init_sewing_master = self.api_robot.Py_Init_SM_Sewing_Master
        self.init_sewing_master.restype = ctypes.c_int

        self.sewing_master = self.api_robot.Py_Write_SM_Sewing_Master
        self.sewing_master.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p)

        # positioning master
        self.init_positioning_master = self.api_robot.Py_Init_SM_Positioning_VS_Master
        self.init_positioning_master.restype = ctypes.c_int

        self.positioning_master = self.api_robot.Py_Write_SM_Positioning_VS_Master
        self.positioning_master.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p)

        self.init_check = self.Init_SM_A()
        if self.init_check == 0:
            print("Robot A Initialization successful")
        else:
            print("Robot A Initialization failed")
        self.init_check = self.Init_SM_B()
        if self.init_check == 0:
            print("Robot B Initialization successful")
        else:
            print("Robot B Initialization failed")
            
        self.init_check = self.init_general_master()
        self.init_check = self.init_sewing_machine_master()
        self.init_check = self.init_grasp_release_master()
        self.init_check = self.init_endeffector_movement_master()
        self.init_check = self.init_fabric_movement_master()
        self.init_check = self.init_sewing_master()
        self.init_check = self.init_positioning_master()

    # get robot information
    def get_robot_angle_A(self):
        curPos_A = np.zeros((6), np.float64)
        self.read_angle_A(curPos_A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        # print(f"Current angle A :{curPos_A}")
        return curPos_A
    
    def get_current_state_A(self):
        curState_A = self.read_state_A()
        # print(f"Current state A :{curState_A}")
        return curState_A
    
    def get_robot_angle_B(self):
        curPos_B = np.zeros((6), np.float64)
        self.read_angle_B(curPos_B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        # print(f"Current angle B :{curPos_B}")
        return curPos_B
    
    def get_current_state_B(self):
        curState_B = self.read_state_B()
        # print(f"Current state B :{curState_B}")
        return curState_B
    
    # general master
    def write_task_state_to_intime(self, task_state):
        self.write_task_state(task_state)
        return
    
    def read_task_init_flag(self):
        task_init_flag = self.read_task_init()
        return task_init_flag
    
    def read_task_fin_flag(self):
        task_fin_flag_A = self.read_task_fin_A()
        task_fin_flag_B = self.read_task_fin_B()
        return task_fin_flag_A, task_fin_flag_B
    
    # sewing machine master
    def wirte_sewing_machine_master(self, task_state, input_value):
        input_double = np.array([float(i) for i in input_value.split()])
        input_double = np.append(input_double, np.zeros(10 - len(input_double)))
        input_double = input_double.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.sewing_machine_master(task_state, input_double)
        return
    
    # grasp and release master
    def wirte_grasp_release_master(self, task_state_A, task_state_B, input_value_A, input_value_B):
        input_value_A = input_value_A.encode('utf-8')
        input_value_B = input_value_B.encode('utf-8')
        self.grasp_release_master(task_state_A, task_state_B, input_value_A, input_value_B)
        return
    
    # end-effector movement master
    def wirte_endeffector_movement_master(self, task_state_A, task_state_B, input_value_A, input_value_B):
        input_value_A = input_value_A.encode('utf-8')
        input_value_B = input_value_B.encode('utf-8')
        self.endeffector_movement_master(task_state_A, task_state_B, input_value_A, input_value_B)
        return
    
    # fabric movement master
    def wirte_fabric_movement_master(self, task_state, input_value_A, input_value_B):
        input_value_A = input_value_A.encode('utf-8')
        input_value_B = input_value_B.encode('utf-8')
        self.fabric_movement_master(task_state, input_value_A, input_value_B)
        return
    
    # sewing master
    def wirte_sewing_master(self, task_state, input_value_A, input_value_B):
        input_value_A = input_value_A.encode('utf-8')
        input_value_B = input_value_B.encode('utf-8')
        self.sewing_master(task_state, input_value_A, input_value_B)
        return
    
    # positioning master
    def wirte_positioning_master(self, task_state, input_value_A, input_value_B):

        input_value_A = input_value_A.encode('utf-8')
        input_value_B = input_value_B.encode('utf-8')
        self.positioning_master(task_state, input_value_A, input_value_B)
        return
    
    # Send command to robot
    def send_command(self, master, arg):
        
        self.init_general_master()
        if master == 1:
            # sewing machine master
            flag = self.init_sewing_machine_master()
            self.wirte_sewing_machine_master(arg[0], arg[1])
            self.write_task_state_to_intime(1)
        elif master == 2:
            # grasp and release master
            flag = self.init_grasp_release_master()
            self.wirte_grasp_release_master(arg[0], arg[1], arg[2], arg[3])
            self.write_task_state_to_intime(2)
        elif master == 3:
            # end-effector movement master
            flag = self.init_endeffector_movement_master()
            self.wirte_endeffector_movement_master(arg[0], arg[1], arg[2], arg[3])
            self.write_task_state_to_intime(3)
        elif master == 4:
            # fabric movement master
            flag = self.init_fabric_movement_master()
            self.wirte_fabric_movement_master(arg[0], arg[1], arg[2])
            self.write_task_state_to_intime(4)
        elif master == 5:
            # fabric positioning visual servoing master
            flag = self.init_positioning_master()
            self.wirte_positioning_master(arg[0], arg[1], arg[2])
            self.write_task_state_to_intime(5)
        elif master == 6:
            # sewing master
            flag = self.init_sewing_master()
            self.wirte_sewing_master(arg[0], arg[1], arg[2])
            self.write_task_state_to_intime(6)
        return

if __name__ == "__main__":

    api_robot = api_robot()
    # api_robot.write_task_state_to_intime(1)
    # api_robot.wirte_sewing_machine_master(1, '1')
    # api_robot.wirte_sewing_machine_master(2, '1')
    # api_robot.wirte_sewing_machine_master(3, '-120 5')

    api_robot.write_task_state_to_intime(2)
    # api_robot.wirte_grasp_release_master(1, 1, ' ', ' ')
    api_robot.wirte_grasp_release_master(2, 2, '3 3', '3 3')
    # api_robot.wirte_grasp_release_master(0, 0, '3 -3', '3 -3')

    # api_robot.write_task_state_to_intime(3)
    # api_robot.wirte_endeffector_movement_master(1, 1, '3 -3', '3 -3')

    # api_robot.write_task_state_to_intime(4)
    # api_robot.wirte_fabric_movement_master(1, '-169.0 -79.0 50.0 0.0 0.0 0.022 1.5 3', '-9.89 -79.0 50.0 0.0 0.0 0.022 3 20')

    # task_state = 0
    # fin_flag = 0
    # while True:
    #     task_state = api_robot.read_task_init_flag()
    #     if task_state == 0:
    #         print("Waiting for task initialization")
    #         time.sleep(1)
    #     if task_state == 1:
    #         print("Task initialized")
    #         break
    # while True:
    #     fin_flag_A, fin_flag_B = api_robot.read_task_fin_flag()
    #     if fin_flag_A == 0 or fin_flag_B == 0:
    #         print("Waiting for task finish")
    #         time.sleep(1)
    #     if fin_flag_A == 1 and fin_flag_B == 1:
    #         print("Task finished")
    #         break




