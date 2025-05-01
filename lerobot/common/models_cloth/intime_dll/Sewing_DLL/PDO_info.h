#ifndef PDO_INFO
#define PDO_INFO

#pragma pack(push, 1)	//構造体のパディング防止

/////////////////////////////////////////////////////////////////
//PDOの書き込み用の先頭アドレス
/////////////////////////////////////////////////////////////////

#define AXIS0_WRITE_CONTROLWORD_ADDR	0
#define AXIS0_WRITE_TARGET_ADDR			2

#define AXIS1_WRITE_CONTROLWORD_ADDR	6
#define AXIS1_WRITE_TARGET_ADDR			8

#define AXIS2_WRITE_CONTROLWORD_ADDR	12
#define AXIS2_WRITE_TARGET_ADDR			14

#define AXIS3_WRITE_CONTROLWORD_ADDR	18
#define AXIS3_WRITE_TARGET_ADDR			20

#define AXIS4_WRITE_CONTROLWORD_ADDR	24
#define AXIS4_WRITE_TARGET_ADDR			26

#define AXIS5_WRITE_CONTROLWORD_ADDR	30
#define AXIS5_WRITE_TARGET_ADDR			32

#define AXIS6_WRITE_CONTROLWORD_ADDR	36
#define AXIS6_WRITE_TARGET_ADDR			38

#define AXIS7_WRITE_CONTROLWORD_ADDR	42
#define AXIS7_WRITE_TARGET_ADDR			44

#define MINI_IO_WRITE_ADDR			48
#define HAND_IO_WRITE_ADDR			50

/////////////////////////////////////////////////////////////////
//PDOの読み取り用の先頭アドレス
/////////////////////////////////////////////////////////////////

#define AXIS0_READ_STATUSWORD_ADDR	0
#define AXIS0_READ_POS_ADDR			2
#define AXIS0_READ_VEL_ADDR			6
#define AXIS0_READ_ELEC_ADDR		10
#define AXIS0_READ_TORQUE_ADDR		12

#define AXIS1_READ_STATUSWORD_ADDR	14
#define AXIS1_READ_POS_ADDR			16
#define AXIS1_READ_VEL_ADDR			20
#define AXIS1_READ_ELEC_ADDR		24
#define AXIS1_READ_TORQUE_ADDR		26

#define AXIS2_READ_STATUSWORD_ADDR	28
#define AXIS2_READ_POS_ADDR			30
#define AXIS2_READ_VEL_ADDR			34
#define AXIS2_READ_ELEC_ADDR		38
#define AXIS2_READ_TORQUE_ADDR		40

#define AXIS3_READ_STATUSWORD_ADDR	42
#define AXIS3_READ_POS_ADDR			44
#define AXIS3_READ_VEL_ADDR			48
#define AXIS3_READ_ELEC_ADDR		52
#define AXIS3_READ_TORQUE_ADDR		54

#define AXIS4_READ_STATUSWORD_ADDR	56
#define AXIS4_READ_POS_ADDR			58
#define AXIS4_READ_VEL_ADDR			62
#define AXIS4_READ_ELEC_ADDR		66
#define AXIS4_READ_TORQUE_ADDR		68

#define AXIS5_READ_STATUSWORD_ADDR	70
#define AXIS5_READ_POS_ADDR			72
#define AXIS5_READ_VEL_ADDR			76
#define AXIS5_READ_ELEC_ADDR		80
#define AXIS5_READ_TORQUE_ADDR		82

#define AXIS6_READ_STATUSWORD_ADDR	84
#define AXIS6_READ_POS_ADDR			86
#define AXIS6_READ_VEL_ADDR			90
#define AXIS6_READ_ELEC_ADDR		94
#define AXIS6_READ_TORQUE_ADDR		96

#define AXIS7_READ_STATUSWORD_ADDR	98
#define AXIS7_READ_POS_ADDR			100
#define AXIS7_READ_VEL_ADDR			104
#define AXIS7_READ_ELEC_ADDR		108
#define AXIS7_READ_TORQUE_ADDR		110

#define MINI_IO_READ_ADDR			112
#define HAND_IO_READ_ADDR			114
#define STATUS_IO_READ_ADDR			115


/////////////////////////////////////////////////////////////////
//PDOの書き込み用の構造体
/////////////////////////////////////////////////////////////////

//PDOの書き込み用軸情報　
typedef struct {
	unsigned short Controlword;
	int desVel;
} Axis_Write_Info;

//PDOの書き込み用の構造体
typedef struct {
	Axis_Write_Info Axis[8];
	unsigned short mini_IO;
	unsigned char hand_IO;
} PDO_W;


/////////////////////////////////////////////////////////////////
//PDOの読み込み用の構造体
/////////////////////////////////////////////////////////////////

//PDOの読み取り用軸情報
typedef struct {
	unsigned short Statusword;
	int actPos;
	int actVel;
	short actElec;
	short actTorque;
} Axis_Read_Info;

//PDOの読み取り用の構造体
typedef struct {
	Axis_Read_Info Axis[8];

	unsigned short mini_IO;
	unsigned char hand_IO;
	unsigned char status_IO;

} PDO_R;

#pragma pack(pop)

#endif	// PDO_INFO
