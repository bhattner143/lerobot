#include "pch.h"
#include "api_INtime.h"

// Standard
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <io.h>
#include <vector>
#include <iostream>
#include <chrono>
#include<tchar.h>

// INtime
#include <ntx.h>
#include "PDO_info.h"
#include "UniversalSM_Exe.h"
#include "./../../../../../K_Controller/K_Controller/SharedMem/SharedMem.h"
#include "./../../../../../K_Controller/K_Controller/Global.h"

#pragma warning(disable:4996)

namespace {
	// instantiate class for shared memory
	static SM_RStatus RobotData_A;
	static SM_Comm Command_A;
	static USM::EXE::Access SM_Comm_Access_A, SM_Save_Access_A, SM_Robot_Access_A;

	static SM_RStatus RobotData_B;
	static SM_Comm Command_B;
	static USM::EXE::Access SM_Comm_Access_B, SM_Save_Access_B, SM_Robot_Access_B;



	sm_general_master SM_General_Master;
	sm_sewing_machine_master SM_Sewing_Machine_Master;
	sm_grasp_release_master SM_Grasp_Release_Master;
	sm_endeffector_movement_master SM_EndEffector_Movement_Master;
	sm_fabric_movement_master SM_Fabric_Movement_Master;
	sm_sewing_master  SM_Sewing_Master;
	sm_positioning_by_vs_master SM_Positioning_VS_Master;
	sm_teleoperate_master SM_Teleoperate_Master;
}


// Shared memory for robot A
int Py_Init_SM_A(void) {

	// find node
	static NTXHANDLE hNodeProcess = ntxGetLocationByName(CONTROL_NODE_1);
	if (hNodeProcess == NTX_BAD_NTXHANDLE) {
		std::cout << "Get Node Process Error" << std::endl;
		getchar();
		return -1;
	}

	// search for root process
	static NTXHANDLE hRootProcess = ntxGetRootRtProcess(hNodeProcess);
	if (hRootProcess == NTX_BAD_NTXHANDLE) {
		std::cout << "Get Root Handle Error" << std::endl;
		getchar();
		return -1;
	}

	// obtain the char instead of const char for the following function requirement
	char* PROCESS_NAME_CHAR = strdup(RPROCESS_NAME);
	char* SM_NAME_SAVE_CHAR = strdup(SM_NAME_SAVE);
	char* SM_NAME_COMM_CHAR = strdup(SM_NAME_COMM);
	char* SM_NAME_ROBOT_CHAR = strdup(SM_NAME_ROBOT);

	// search for controller process
	NTXHANDLE hContProcess = ntxLookupNtxHandle(hRootProcess, PROCESS_NAME_CHAR, WAIT_FOREVER);
	if (hContProcess == NTX_BAD_NTXHANDLE) {
		std::cout << "Controller Process Handle Error" << std::endl;
		getchar();
		return -1;
	}

	// search shared memory for storage
	NTXHANDLE hSM_Save = ntxLookupNtxHandle(hContProcess, SM_NAME_SAVE_CHAR, WAIT_FOREVER);
	if (hSM_Save == NTX_BAD_NTXHANDLE) {
		std::cout << "SaveData Shared Memory Handle Error" << std::endl;
		getchar();
		return -1;
	}

	// command shared memory retrieval
	NTXHANDLE hSM_Comm = ntxLookupNtxHandle(hContProcess, SM_NAME_COMM_CHAR, WAIT_FOREVER);
	if (hSM_Comm == NTX_BAD_NTXHANDLE) {
		std::cout << "Command Shared Memory Handle Error" << std::endl;
		getchar();
		return -1;
	}

	// search shared memory for reading robot state
	NTXHANDLE hSM_Robot = ntxLookupNtxHandle(hContProcess, SM_NAME_ROBOT_CHAR, WAIT_FOREVER);
	if (hSM_Robot == NTX_BAD_NTXHANDLE) {
		std::cout << "Robot Shared Memory Handle Error" << std::endl;
		getchar();
		return -1;
	}

	SM_Comm_Access_A.SetHandleEx(hSM_Comm, NTX_MAP_UNALIGNED);
	SM_Robot_Access_A.SetHandleEx(hSM_Robot, NTX_MAP_UNALIGNED);
	SM_Save_Access_A.SetHandle(hSM_Save);

	// Initialize shared memory
	memset(&RobotData_A, 0, sizeof(RobotData_A));
	memset(&Command_A, 0, sizeof(Command_A));

	return 0;
}

void Py_Read_SM_A(double* PyCurPos)
{
	// Read current pose of Robot A
	SM_Comm_Access_A.Read((unsigned char*)&Command_A, 0, sizeof(Command_A));
	SM_Robot_Access_A.Read((unsigned char*)&RobotData_A, 0, sizeof(RobotData_A));
	for (int axis = 0; axis < AXIS_NUM; axis++) {
		PyCurPos[axis] = RobotData_A.curPos[axis];
	}
}

int Py_Read_Status_A(void)
{
	// Read current state
	SM_Comm_Access_A.Read((unsigned char*)&Command_A, 0, sizeof(Command_A));
	SM_Robot_Access_A.Read((unsigned char*)&RobotData_A, 0, sizeof(RobotData_A));
	return Command_A.nowState;
}

// Shared memory for robot B
int Py_Init_SM_B(void) {

	// find node
	static NTXHANDLE hNodeProcess = ntxGetLocationByName(CONTROL_NODE_2);
	if (hNodeProcess == NTX_BAD_NTXHANDLE) {
		std::cout << "Get Node Process Error" << std::endl;
		getchar();
		return -1;
	}

	// search for root process
	static NTXHANDLE hRootProcess = ntxGetRootRtProcess(hNodeProcess);
	if (hRootProcess == NTX_BAD_NTXHANDLE) {
		std::cout << "Get Root Handle Error" << std::endl;
		getchar();
		return -1;
	}

	// obtain the char instead of const char for the following function requirement
	char* PROCESS_NAME_CHAR = strdup(RPROCESS_NAME);
	char* SM_NAME_SAVE_CHAR = strdup(SM_NAME_SAVE);
	char* SM_NAME_COMM_CHAR = strdup(SM_NAME_COMM);
	char* SM_NAME_ROBOT_CHAR = strdup(SM_NAME_ROBOT);

	// search for controller process
	NTXHANDLE hContProcess = ntxLookupNtxHandle(hRootProcess, PROCESS_NAME_CHAR, WAIT_FOREVER);
	if (hContProcess == NTX_BAD_NTXHANDLE) {
		std::cout << "Controller Process Handle Error" << std::endl;
		getchar();
		return -1;
	}

	// search shared memory for storage (use for task control not for save cennect with dll for send command to intime)
	NTXHANDLE hSM_Save = ntxLookupNtxHandle(hContProcess, SM_NAME_SAVE_CHAR, WAIT_FOREVER);
	if (hSM_Save == NTX_BAD_NTXHANDLE) {
		std::cout << "SaveData Shared Memory Handle Error" << std::endl;
		getchar();
		return -1;
	}

	// command shared memory retrieval (store command from traj to controller)
	NTXHANDLE hSM_Comm = ntxLookupNtxHandle(hContProcess, SM_NAME_COMM_CHAR, WAIT_FOREVER);
	if (hSM_Comm == NTX_BAD_NTXHANDLE) {
		std::cout << "Command Shared Memory Handle Error" << std::endl;
		getchar();
		return -1;
	}

	// search shared memory for reading robot state (store robot information)
	NTXHANDLE hSM_Robot = ntxLookupNtxHandle(hContProcess, SM_NAME_ROBOT_CHAR, WAIT_FOREVER);
	if (hSM_Robot == NTX_BAD_NTXHANDLE) {
		std::cout << "Robot Shared Memory Handle Error" << std::endl;
		getchar();
		return -1;
	}

	SM_Comm_Access_B.SetHandleEx(hSM_Comm, NTX_MAP_UNALIGNED);
	SM_Robot_Access_B.SetHandleEx(hSM_Robot, NTX_MAP_UNALIGNED);
	SM_Save_Access_B.SetHandle(hSM_Save);

	// Initialize shared memory
	memset(&RobotData_B, 0, sizeof(RobotData_B));
	memset(&Command_B, 0, sizeof(Command_B));

	return 0;
}

void Py_Read_SM_B(double* PyCurPos)
{
	// Read current pose of Robot B
	SM_Comm_Access_B.Read((unsigned char*)&Command_B, 0, sizeof(Command_B));
	SM_Robot_Access_B.Read((unsigned char*)&RobotData_B, 0, sizeof(RobotData_B));
	for (int axis = 0; axis < AXIS_NUM; axis++) {
		PyCurPos[axis] = RobotData_B.curPos[axis];
	}
}

int Py_Read_Status_B(void)
{
	// Read current state
	SM_Comm_Access_B.Read((unsigned char*)&Command_B, 0, sizeof(Command_B));
	SM_Robot_Access_B.Read((unsigned char*)&RobotData_B, 0, sizeof(RobotData_B));
	return Command_B.nowState;
}

// Shared memory for general master
int Py_Init_SM_General_Master(void) {

	memset(&SM_General_Master, 0, sizeof(SM_General_Master));
	SM_Save_Access_A.Write((unsigned char*)&SM_General_Master, addr_general_master, sizeof(SM_General_Master));

	return 1;
}

void Py_Write_SM_General_Master(int name_task) {
	
	SM_General_Master.task_flag_w2i = name_task;
	SM_Save_Access_A.Write((unsigned char*)&SM_General_Master, addr_general_master, sizeof(SM_General_Master));

	return;
}

int Py_Read_General_Task_Init_Flag(void) {
	
	SM_Save_Access_A.Read((unsigned char*)&SM_General_Master, addr_general_master, sizeof(SM_General_Master));

	return SM_General_Master.init_flag_i2w;
}

int Py_Read_General_Task_End_Flag_A(void) {

	SM_Save_Access_A.Read((unsigned char*)&SM_General_Master, addr_general_master, sizeof(SM_General_Master));

	return SM_General_Master.fin_flag_A_i2w;
}

int Py_Read_General_Task_End_Flag_B(void) {

	SM_Save_Access_A.Read((unsigned char*)&SM_General_Master, addr_general_master, sizeof(SM_General_Master));

	return SM_General_Master.fin_flag_B_i2w;
}

// Shared memory for sewing machine master
int Py_Init_SM_Sewing_Machine_Master(void) {

	memset(&SM_Sewing_Machine_Master, 0, sizeof(SM_Sewing_Machine_Master));
	SM_Save_Access_A.Write((unsigned char*)&SM_Sewing_Machine_Master, addr_sewing_machine_master, sizeof(SM_Sewing_Machine_Master));

	return 1;
}

void Py_Write_SM_Sewing_Machine_Master(int name_task, double inputVal[10]) {
	
	memcpy(SM_Sewing_Machine_Master.inputVal, inputVal, sizeof(SM_Sewing_Machine_Master.inputVal));
	
	if (name_task == 1) {
		SM_Sewing_Machine_Master.flag_pressor_foot_control = 1;
	}
	else if (name_task == 2) {
		SM_Sewing_Machine_Master.flag_thread_cutting = 1;
	}
	else if (name_task == 3) {
		SM_Sewing_Machine_Master.flag_needle_penetrate = 1;
	}
	else {
		memset(&SM_Sewing_Machine_Master, 0, sizeof(SM_Sewing_Machine_Master));
	}
	SM_Save_Access_A.Write((unsigned char*)&SM_Sewing_Machine_Master, addr_sewing_machine_master, sizeof(SM_Sewing_Machine_Master));

	std::cout << "I am in c++ sewing machine control Buffer A contains: " << inputVal << std::endl;

	return;
}

// Shared memory for grasp release master
int Py_Init_SM_Grasp_Release_Master(void) {

	memset(&SM_Grasp_Release_Master, 0, sizeof(SM_Grasp_Release_Master));
	SM_Save_Access_A.Write((unsigned char*)&SM_Grasp_Release_Master, addr_grasp_release_master, sizeof(SM_Grasp_Release_Master));

	return 1;
}

void Py_Write_SM_Grasp_Release_Master(int task_A, int task_B, char inputVal_A[1024], char inputVal_B[1024]) {

	memcpy(SM_Grasp_Release_Master.inputVal_A, inputVal_A, sizeof(SM_Grasp_Release_Master.inputVal_A));
	memcpy(SM_Grasp_Release_Master.inputVal_B, inputVal_B, sizeof(SM_Grasp_Release_Master.inputVal_B));

	if (task_A == 0 || task_A == 1 || task_A == 2 || task_A == 3){
		SM_Grasp_Release_Master.task_A = task_A;
	}
	else {
		SM_Grasp_Release_Master.task_A = 0;
	}

	if (task_B == 0 || task_B == 1 || task_B == 2 || task_B == 3) {
		SM_Grasp_Release_Master.task_B = task_B;
	}
	else {
		SM_Grasp_Release_Master.task_B = 0;
	}

	SM_Save_Access_A.Write((unsigned char*)&SM_Grasp_Release_Master, addr_grasp_release_master, sizeof(SM_Grasp_Release_Master));

	std::cout << "I am in c++ grasp release Buffer A contains: " << inputVal_A << std::endl;
	std::cout << "I am in c++ grasp release Buffer B contains: " << inputVal_B << std::endl;

	return;
}


// Shared memory for end effector movement master
int Py_Init_SM_EndEffector_Movement_Master(void) {

	memset(&SM_EndEffector_Movement_Master, 0, sizeof(SM_EndEffector_Movement_Master));
	SM_Save_Access_A.Write((unsigned char*)&SM_EndEffector_Movement_Master, addr_end_effector_movement_master, sizeof(SM_EndEffector_Movement_Master));

	return 1;
}

void Py_Write_SM_EndEffector_Movement_Master(int task_A, int task_B, char inputVal_A[1024], char inputVal_B[1024]) {

	memcpy(SM_EndEffector_Movement_Master.inputVal_A, inputVal_A, sizeof(SM_EndEffector_Movement_Master.inputVal_A));
	memcpy(SM_EndEffector_Movement_Master.inputVal_B, inputVal_B, sizeof(SM_EndEffector_Movement_Master.inputVal_B));

	if (task_A == 0 || task_A == 1) {
		SM_EndEffector_Movement_Master.endeffector_A = task_A;
	}
	else {
		SM_EndEffector_Movement_Master.endeffector_A = 0;
	}

	if (task_B == 0 || task_B == 1) {
		SM_EndEffector_Movement_Master.endeffector_B = task_B;
	}
	else {
		SM_EndEffector_Movement_Master.endeffector_B = 0;
	}

	SM_Save_Access_A.Write((unsigned char*)&SM_EndEffector_Movement_Master, addr_end_effector_movement_master, sizeof(SM_EndEffector_Movement_Master));

	std::cout << "I am in c++ endeffector movement Buffer A contains: " << inputVal_A << std::endl;
	std::cout << "I am in c++ endeffector movement Buffer B contains: " << inputVal_B << std::endl;

	return;
}

// Shared memory for fabric movement master
int Py_Init_SM_Fabric_Movement_Master(void) {

	memset(&SM_Fabric_Movement_Master, 0, sizeof(SM_Fabric_Movement_Master));
	SM_Save_Access_A.Write((unsigned char*)&SM_Fabric_Movement_Master, addr_fabric_movement_master, sizeof(SM_Fabric_Movement_Master));

	return 1;
}

void Py_Write_SM_Fabric_Movement_Master(int task, char inputVal_A[1024], char inputVal_B[1024]) {
	memcpy(SM_Fabric_Movement_Master.inputVal_A, inputVal_A, sizeof(SM_Fabric_Movement_Master.inputVal_A));
	memcpy(SM_Fabric_Movement_Master.inputVal_B, inputVal_B, sizeof(SM_Fabric_Movement_Master.inputVal_B));

	if (task == 1 || task == 2) {
		SM_Fabric_Movement_Master.name_task = task;
	}
	else {
		SM_Fabric_Movement_Master.name_task = 0;
	}
	SM_Save_Access_A.Write((unsigned char*)&SM_Fabric_Movement_Master, addr_fabric_movement_master, sizeof(SM_Fabric_Movement_Master));

	std::cout << "I am in c++ fabric movement Buffer A contains: " << inputVal_A << std::endl;
	std::cout << "I am in c++ fabric movement Buffer B contains: " << inputVal_B << std::endl;

}


// Shared memory for sewing master
int Py_Init_SM_Sewing_Master(void) {

	memset(&SM_Sewing_Master, 0, sizeof(SM_Sewing_Master));
	SM_Save_Access_A.Write((unsigned char*)&SM_Sewing_Master, addr_sewing_master, sizeof(SM_Sewing_Master));

	return 1;
}

void Py_Write_SM_Sewing_Master(int task, char inputVal_A[1024], char inputVal_B[1024]) {
	memcpy(SM_Sewing_Master.inputVal_A, inputVal_A, sizeof(SM_Sewing_Master.inputVal_A));
	memcpy(SM_Sewing_Master.inputVal_B, inputVal_B, sizeof(SM_Sewing_Master.inputVal_B));

	if (task == 1) {
		SM_Sewing_Master.name_task = task;
	}
	else {
		SM_Sewing_Master.name_task = 0;
	}
	SM_Save_Access_A.Write((unsigned char*)&SM_Sewing_Master, addr_sewing_master, sizeof(SM_Sewing_Master));

	std::cout << "I am in c++ sewing Buffer A contains: " << inputVal_A << std::endl;
	std::cout << "I am in c++ sewing Buffer B contains: " << inputVal_B << std::endl;
}

// Shared memory for positioning vs master
int Py_Init_SM_Positioning_VS_Master(void) {

	memset(&SM_Positioning_VS_Master, 0, sizeof(SM_Positioning_VS_Master));
	SM_Save_Access_A.Write((unsigned char*)&SM_Positioning_VS_Master, addr_fabric_positioning_master, sizeof(SM_Positioning_VS_Master));

	return 1;
}

void Py_Write_SM_Positioning_VS_Master(int task, char inputVal_A[1024], char inputVal_B[1024]) {
	memcpy(SM_Positioning_VS_Master.inputVal_A, inputVal_A, sizeof(SM_Positioning_VS_Master.inputVal_A));
	memcpy(SM_Positioning_VS_Master.inputVal_B, inputVal_B, sizeof(SM_Positioning_VS_Master.inputVal_B));

	if (task == 1 || task == 2) {
		SM_Positioning_VS_Master.name_task = task;
	}
	else {
		SM_Positioning_VS_Master.name_task = 0;
	}
	SM_Save_Access_A.Write((unsigned char*)&SM_Positioning_VS_Master, addr_fabric_positioning_master, sizeof(SM_Positioning_VS_Master));

	std::cout << "I am in c++ Positioning_VS Buffer A contains: " << inputVal_A << std::endl;
	std::cout << "I am in c++ Positioning_VS Buffer B contains: " << inputVal_B << std::endl;
}

// Shared memory for teleoperate master
int Py_Init_SM_Teleoperate_Master(void) {

	memset(&SM_Teleoperate_Master, 0, sizeof(SM_Teleoperate_Master));
	SM_Save_Access_A.Write((unsigned char*)&SM_Teleoperate_Master, addr_teleoperate_master, sizeof(SM_Teleoperate_Master));

	return 1;
}

void Py_Write_SM_Teleoperate_Master(int task, char inputVal_A[1024], char inputVal_B[1024]) {
	memcpy(SM_Teleoperate_Master.inputVal_A, inputVal_A, sizeof(SM_Teleoperate_Master.inputVal_A));
	memcpy(SM_Teleoperate_Master.inputVal_B, inputVal_B, sizeof(SM_Teleoperate_Master.inputVal_B));

	if (task == 1) {
		SM_Teleoperate_Master.name_task = task;
	}
	else {
		SM_Teleoperate_Master.name_task = 0;
	}
	SM_Save_Access_A.Write((unsigned char*)&SM_Teleoperate_Master, addr_sewing_master, sizeof(SM_Teleoperate_Master));

	std::cout << "I am in c++ sewing Buffer A contains: " << inputVal_A << std::endl;
	std::cout << "I am in c++ sewing Buffer B contains: " << inputVal_B << std::endl;
}