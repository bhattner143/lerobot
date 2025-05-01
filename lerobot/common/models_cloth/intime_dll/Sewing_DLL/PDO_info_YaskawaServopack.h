#ifndef PDO_INFO_YASKAWASERVOPACK
#define PDO_INFO_YASKAWASERVOPACK

#pragma pack(push, 1)	//Prevention of structure padding

//SERVOPACK operation mode is set to Cyclic sync velocity mode

/////////////////////////////////////////////////////////////////
//First address for writing PDO
/////////////////////////////////////////////////////////////////
// 
//Write-PDO: 0. Controlword (UINT), 1. Target Velocity (DINT), 2. Digital Outputs (UDINT), 3. Digital Outputs Bitmask (UDINT)
// 
#define YASKAWA_SERVOPACK_WRITE_CONTROLWORD_ADDR	0
#define YASKAWA_SERVOPACK_WRITE_TARGETVEL_ADDR		2
#define YASKAWA_SERVOPACK_WRITE_DO_ADDR				6 //Input to DIO board
#define YASKAWA_SERVOPACK_WRITE_DOBITMASK_ADDR		10 //Input to DIO board

/////////////////////////////////////////////////////////////////
//First address for reading PDO
/////////////////////////////////////////////////////////////////
// 
//Read-PDO: 0. Statusword (UINT), 1. Position Actual Value (DINT), 2. Velocity Actual Value (DINT), 3. Torque Actual Value (INT)
//
#define YASKAWA_SERVOPACK_READ_STATUSWORD_ADDR	0
#define YASKAWA_SERVOPACK_READ_ACTUALPOS_ADDR	2
#define YASKAWA_SERVOPACK_READ_ACTUALVEL_ADDR	6
#define YASKAWA_SERVOPACK_READ_ACTUALTRQ_ADDR	10
#define YASKAWA_SERVOPACK_READ_DOOUTPUT_ADDR	14 //Output from DIO board

/////////////////////////////////////////////////////////////////
//Structure for writing PDO
/////////////////////////////////////////////////////////////////

//Axis information for writing PDOÅ@
typedef struct {
	unsigned short Controlword;
	int desVel;
	unsigned int DigitalOutputs;// = 0x00000000;// All channels: OFF (bits 17, 18, 19: 0)
	unsigned int DO_Bitmask;// = 0x000E0000;//All channels: Enable (bits 17, 18, 19: 1)
} Yaskawa_Axis_Write_Info;

//Structure for writing PDO
typedef struct {
	Yaskawa_Axis_Write_Info Axis[3];
} PDO_YaskawaServopack_W;


/////////////////////////////////////////////////////////////////
//Structure for reading PDO
/////////////////////////////////////////////////////////////////

//Axis information for reading PDO
typedef struct {
	unsigned short Statusword;
	int actPos;
	int actVel;
	short actTrq;
	//unsigned int DigitalInputs;
} Yaskawa_Axis_Read_Info;

//Structure for reading PDO
typedef struct {
	Yaskawa_Axis_Read_Info Axis[3];
} PDO_YaskawaServopack_R;

#pragma pack(pop)

#endif	// PDO_INFO_FTSENSOR
