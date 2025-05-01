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

namespace {
	// instantiate class for shared memory
	static SM_RStatus RobotData_A;
	static SM_Comm Command_A;
	static USM::EXE::Access SM_Comm_Access_A, SM_Save_Access_A, SM_Robot_Access_A;

	static SM_RStatus RobotData_B;
	static SM_Comm Command_B;
	static USM::EXE::Access SM_Comm_Access_B, SM_Save_Access_B, SM_Robot_Access_B;

	DualLoadingAlignment Dual_Loading_Alignment;
}

// Shared memory for robot A
int Py_Init_SM_A_Positioning(void) {

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
	char* PROCESS_NAME_CHAR = _strdup(RPROCESS_NAME);
	char* SM_NAME_SAVE_CHAR = _strdup(SM_NAME_SAVE);
	char* SM_NAME_COMM_CHAR = _strdup(SM_NAME_COMM);
	char* SM_NAME_ROBOT_CHAR = _strdup(SM_NAME_ROBOT);

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
	SM_Save_Access_A.SetHandle(hSM_Save);

	return 0;
}

// Shared memory for robot B
int Py_Init_SM_B_Positioning(void) {

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
	char* PROCESS_NAME_CHAR = _strdup(RPROCESS_NAME);
	char* SM_NAME_SAVE_CHAR = _strdup(SM_NAME_SAVE);
	char* SM_NAME_COMM_CHAR = _strdup(SM_NAME_COMM);
	char* SM_NAME_ROBOT_CHAR = _strdup(SM_NAME_ROBOT);

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
	SM_Save_Access_B.SetHandle(hSM_Save);

	return 0;
}

// Shared memory for loading alignment
State_Loading Py_Read_Loading_Alignment_State(double* PyCurEndPos_A, double* PyCurEndPos_B)
{
	State_Loading states;
	// Read current pose of Robot A
	SM_Comm_Access_A.Read((unsigned char*)&Command_A, 0, sizeof(Command_A));

	// Read current pose of Robot B
	SM_Comm_Access_B.Read((unsigned char*)&Command_B, 0, sizeof(Command_B));

	// Read visual servoing shared memory
	SM_Save_Access_B.Read((unsigned char*)&Dual_Loading_Alignment, addr_positioning_w2i, sizeof(Dual_Loading_Alignment));
	for (int axis = 0; axis < AXIS_NUM; axis++) {
		PyCurEndPos_A[axis] = Dual_Loading_Alignment.Toolend_Pose_WorldOrient_A[axis];
	}
	for (int axis = 0; axis < AXIS_NUM; axis++) {
		PyCurEndPos_B[axis] = Dual_Loading_Alignment.Toolend_Pose_WorldOrient_B[axis];
	}

	states.state_A = Command_A.nowState;
	states.state_B = Command_B.nowState;
	states.state_vs = Dual_Loading_Alignment.state_vs;

	return states;
}

int Py_Write_Loading_Alignment(double v_x_A, double v_y_A, double v_tz_A, double v_x_B, double v_y_B, double v_tz_B, bool finFlag) {

	SM_Save_Access_B.Read((unsigned char*)&Dual_Loading_Alignment, addr_positioning_w2i, sizeof(Dual_Loading_Alignment));

	for (int i = 0; i < 6; i++) {
		Dual_Loading_Alignment.Toolend_velocity_A[i] = 0.0;
		Dual_Loading_Alignment.Toolend_velocity_B[i] = 0.0;
	}

	Dual_Loading_Alignment.Toolend_velocity_A[0] = v_x_A;
	Dual_Loading_Alignment.Toolend_velocity_A[1] = v_y_A;
	Dual_Loading_Alignment.Toolend_velocity_A[5] = v_tz_A;
	Dual_Loading_Alignment.Toolend_velocity_B[0] = v_x_B;
	Dual_Loading_Alignment.Toolend_velocity_B[1] = v_y_B;
	Dual_Loading_Alignment.Toolend_velocity_B[5] = v_tz_B;
	Dual_Loading_Alignment.finFlag = finFlag;
	SM_Save_Access_B.Write((unsigned char*)&Dual_Loading_Alignment, addr_positioning_w2i, sizeof(Dual_Loading_Alignment));

	return 1;
}
