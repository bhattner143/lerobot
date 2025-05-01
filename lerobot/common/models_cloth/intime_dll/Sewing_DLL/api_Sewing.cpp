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

	DualSewingFeedback_w2i Dual_Sewing_Feedback;
	DualSewingInfo_i2w Dual_Sewing_Info;
}

// Shared memory for robot A
int Py_Init_SM_A_Sewing(void) {

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
int Py_Init_SM_B_Sewing(void) {

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

// Shared memory for sewing feedback
int Py_Init_SM_SewingFeedback(void) {

	memset(&Dual_Sewing_Feedback, 0, sizeof(Dual_Sewing_Feedback));
	Dual_Sewing_Feedback.desired_stitch = 30;
	Dual_Sewing_Feedback.sewFlag = 0;
	Dual_Sewing_Feedback.finFlag = false;
	Dual_Sewing_Feedback.rotation_speed = 0;

	return 1;
}

int Py_Write_Sewing_Feedback(double Rotation_Speed, int desired_stitch, bool finFlag, unsigned int sewFlag) {

	Dual_Sewing_Feedback.rotation_speed = Rotation_Speed;
	Dual_Sewing_Feedback.finFlag = finFlag;
	Dual_Sewing_Feedback.desired_stitch = desired_stitch;
	Dual_Sewing_Feedback.sewFlag = sewFlag;
	SM_Save_Access_B.Write((unsigned char*)&Dual_Sewing_Feedback, addr_sewing_feedback_w2i, sizeof(Dual_Sewing_Feedback));

	return 1;
}

double Py_Read_Sewing_Speed(void) {
	// Read current pose of Robot B
	SM_Comm_Access_B.Read((unsigned char*)&Command_B, 0, sizeof(Command_B));
	return Command_B.sewingSpeed;
}

State_Sewing Py_Read_Sewing_State(void) {
	State_Sewing states;
	// Read current pose of Robot A
	SM_Comm_Access_A.Read((unsigned char*)&Command_A, 0, sizeof(Command_A));
	// Read current pose of Robot B
	SM_Comm_Access_B.Read((unsigned char*)&Command_B, 0, sizeof(Command_B));
	states.state_A = Command_A.nowState;
	states.state_B = Command_B.nowState;
	return states;
}

int Py_Read_Final_Sewn_Stitches(void) {
	// Read current pose of Robot B
	SM_Save_Access_B.Read((unsigned char*)&Dual_Sewing_Info, addr_sewing_information_i2w, sizeof(Dual_Sewing_Info));
	return Dual_Sewing_Info.sewn_stitch;
}
