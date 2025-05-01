#ifndef SHAREDMEM_H
#define SHAREDMEM_H

#include "PDO_info.h"
#include "PDO_info_FTsensor.h"
#include "PDO_info_YaskawaServopack.h"
#include "iostream"

using namespace std;

#define SM_SIZE_ROBOT		4096		//Share memory size
#define SM_SIZE_COMM		4096		//Share memory size
#define SM_SIZE_HAND        4096        //Share memory size
#define SM_SIZE_TRJ_LINK	4096        //Share memory size
#define SM_SIZE_FTSENSOR	4096		//Share memory size
#define SM_SIZE_YASKAWASERVOPACK	4096		//Share memory size

#define SM_POINT_SIZE       1100

///////////////////////////////////////////////////////////////////////////////////
// Set information to be sent in shared memory
// Change here if you want to change the contents of shared memory
///////////////////////////////////////////////////////////////////////////////////
typedef struct {
	bool initialFlag;
	bool finFlag;
	PDO_W PDO_write;
	int nowState;
	double desPos[6];
	double desVel[6];
	double LastError[6]; // for PD control
	double desTEPos_Trj[6];
	double ExternalForce_WorldCo[6];
	double ObjExF[6]; //Current position and posture of the object
	double tgtObjPose[6]; //Position based on target orbit
	double ObjExternalForce[6]; //Current external force on the object
	double curIntForce[6]; //Present internal force on an object
	double motorAngle;
	double sewingSpeed;
	double rotationSpeed;
	double real_sewingSpeed;
	
} SM_Comm;

typedef struct {
	bool initialFlag;
	bool finFlag;
	PDO_R PDO_read;
	unsigned long int loop;
	double curPos[6];
	double curVel[6];
	int syncFlag;
	chrono::system_clock::time_point startTime;
} SM_RStatus;


typedef enum {
	HAND_TORQUE = 0,
	HAND_RESET,
	HAND_VOLTAGE,
	HAND_POSITION,
	ESG2_OPEN,
	ESG2_CLOSE,
	SUCTION_ON,
	SUCTION_OFF,
	ROLLINGUP
} HAND_CONTROLMODE;


///////////////////////////////////////////////////////////////////////////////////

#pragma pack(push, 1)
typedef struct {
	HAND_CONTROLMODE controlMode;
	double desVol;
	double desPos;
	double desVel;
	double desAcc;
} SM_HandCommand;

typedef struct {
	double curPos;
	double curVel;
	double curAcc;

	double curVol;
	int    ESG2HoldFlg;
} SM_HandStatus;

typedef struct {
	SM_HandCommand HandComm;
	SM_HandStatus HandStatus;

} SM_HandInfo;
#pragma pack(pop)

///////////////////////////////////////////////////////////////////////////////////
// --------------------- Force and Torque Sensor ---------------------------

#pragma pack(push, 1)
typedef struct {
	double Fx;
	double Fy;
	double Fz;
	double Tx;
	double Ty;
	double Tz;
}ForceTorque;

typedef struct {
	PDO_FT_W PDO_FT_write;
	PDO_FT_R PDO_FT_read;
	ForceTorque SensorFT;
	ForceTorque ToolEndFT;
	ForceTorque ExternalFT;
	double curInternalFT;
	double desInternalFT;
	bool biasResetFlg;

}SM_FTsensorInfo;

#pragma pack(pop)

///////////////////////////////////////////////////////////////////////////////////

#pragma pack(push, 1)
typedef struct {
	PDO_YaskawaServopack_W PDO_YaskawaServopack_write;
	int nowState;
	double desPos[3];
	double desVel[3];
}SM_YaskawaServopackCommand;

typedef struct {
	PDO_YaskawaServopack_R PDO_YaskawaServopack_read;
	unsigned long int loop;
	double curPos[3];
	double curVel[3];
}SM_YaskawaServopackStatus;

typedef struct {
	SM_YaskawaServopackCommand YaskawaServopackComm;
	SM_YaskawaServopackStatus  YaskawaServopackStatus;
}SM_YaskawaServopackInfo;
#pragma pack(pop)

typedef struct {
	double desPos[6];
	double desVel[6];
}SM_TrjComm_DEBUG;

#endif //SHAREDMEM_H