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

	TeleoperateComm_w2i Tele_Comm;
}

// Shared memory for robot A
int Py_Init_SM_Teleoperate_A(void) {

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
int Py_Init_SM_Teleoperate_B(void) {

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

int Py_Init_SM_Teleoperation(void) {

	memset(&Tele_Comm, 0, sizeof(Tele_Comm));

	return 1;
}

void Py_Teleoperation_w2i(double DesFT_A[6], double DesFT_B[6]) {

	SM_Save_Access_A.Read((unsigned char*)&Tele_Comm, addr_teleoperate_w2i, sizeof(Tele_Comm));

	memcpy(Tele_Comm.desFT_TEWorld_A_w2i, DesFT_A, sizeof(Tele_Comm.desFT_TEWorld_A_w2i));
	memcpy(Tele_Comm.desFT_TEWorld_B_w2i, DesFT_B, sizeof(Tele_Comm.desFT_TEWorld_B_w2i));

	//std::cout << Tele_Comm.desFT_TEWorld_A_w2i[0] << std::endl;
	//std::cout << Tele_Comm.desFT_TEWorld_B_w2i[0] << std::endl;

	SM_Save_Access_A.Write((unsigned char*)&Tele_Comm, addr_teleoperate_w2i, sizeof(Tele_Comm));

	return;
}
