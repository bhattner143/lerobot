#ifndef GLOBAL_H
#define GLOBAL_H

/////////////////////////////////////////////////////////////////////////////////////////
// Basic Parameter Setting
/////////////////////////////////////////////////////////////////////////////////////////
#ifndef HIGH
#define HIGH			1
#endif

#ifndef LOW
#define LOW				0
#endif

#ifndef ERROR
#define ERROR			-1
#endif

#ifndef PI
#define PI				3.14159265358979323846
#endif 

#ifndef AXIS_NUM
#define AXIS_NUM		6
#endif

#ifndef YASKAWA_AXIS_NUM
#define YASKAWA_AXIS_NUM		1
#endif

#ifndef DEG2RAD
#define DEG2RAD			PI / 180.0
#endif

#ifndef RAD2DEG
#define RAD2DEG			180.0 / PI
#endif

// Define for the hanged robot
#define TENTURI_ROBOT

//Define if force sensor is not used
//#define FORCE_SENSOR_OFF_MODE

//Define if hand is not used
#define HAND_OFF_MODE

//Check if servo motor for sewing machine is not used.
//#define YASKAWA_OFF_MODE

// Used when encoder pulse information is used
const double ENC_RESOLUTION[AXIS_NUM] = { 131072.0, 131072.0 , 131072.0 , 131072.0 , 131072.0 , 131072.0 };

const double ENC_DIR[AXIS_NUM] = { -1.0, 1.0, 1.0, -1.0, 1.0, -1.0 };

const double GEAR_RATIO[AXIS_NUM] = { 80.0, 100.0, 80.0, 80.0, 3434.0 / 43.0, 50.0 };

//2022/12/29 Hong Kong 3D Sewing System CALSET (facing to sewing machine (switch Node), left is A, right is B)
const double CALSET_1[AXIS_NUM] = { -110129, 50069, -75993, -57615, -124961, 519682 }; //Calset value for RobotA
const double CALSET_2[AXIS_NUM] = { -22172, -162205, 17023, -31164, -35537, -1925507 }; //Calset value for RobotB

// Used only when dealing with angle information
#define ENC2DEG	1.0 / 65536.0

#define ENC2RAD ENC2DEG * DEG2RAD

#define DEG2ENC	65536.0

#define RAD2ENC	DEG2ENC * DEG2RAD

/////////////////////////////////////////////////////////////////////////////////////////
//Name of public object
/////////////////////////////////////////////////////////////////////////////////////////

#define SM_NAME_ROBOT	"Sharedmem" //Memory for writing the current robot state

#define SM_NAME_COMM	"Sharedmem2" //Memory for writing current state 

#define SM_NAME_HAND	"Sharedmem3" //Shared memory for hand

#define SM_NAME_LOG		"Sharedmemlog" //Shared memory for log storage

#define SM_NAME_SAVE	"Sharedmem4" //Memory for writing to specified memory locations that each individual wishes to use

#define SM_NAME_TRJ_LINK	"Sharedmem5" //Shared memory for GUI commands

#define SM_NAME_FTSENSOR	"Sharedmem6" //Memory for writing force sensor values

#define SM_NAME_YASKAWASERVOPACK	"Sharedmem7" //Memory for writing Yaskawa servo motor commands

#define ROBOT_SEM_NAME	"RobotSemA" //Semaphore for robot state common memory

#define HAND_SEM_NAME	"HandSem" //Semaphore for hand shared memory

#define FTSENSOR_SEM_NAME	"FTsensorSem" //Semaphore for force sensor shared memory

#define YASKAWASERVOPACK_SEM_NAME	"YaskawaSVSem" //Semaphore for Yaskawa servo motor shared memory

#define RPROCESS_NAME	"ProcContRT" //controller process

#define	NAME_MAILCONT	"MailCont"

#define NAME_MAILBOX_TRAJ	"MailTraj"

#define	NAME_MAIL_SIM_STATE	"MailSimState"

/////////////////////////////////////////////////////////////////////////////////////////
//Robot parameter settings
/////////////////////////////////////////////////////////////////////////////////////////

#define CONTROL_NODE_1	"NodeA"

#define CONTROL_NODE_2	"NodeB"

#define CONTROL_NODE_3	"NodeS"

#define TRAJECTORY_NODE_1	"TrajectoryNodeA"

#define TRAJECTORY_NODE_2	"TrajectoryNodeB"

#define TRAJECTORY_NODE_3	"TrajectoryNodeD"

#define CONT_TICK_US	250.0	// Operating time per cycle [us]

#define CONT_TICK_MS	CONT_TICK_US / 1000.0	// Operating time per cycle [ms]

#define FREQ			4000	// Operating Frequency

#define SERVO_OFF		0

#define SERVO_READY		6

#define SERVO_ON		15

#define SERVO_SHUTDOWN	2

enum SERVOSTATE_VALUE
{
	SS_NOT_READY = 1,
	SS_SWITCH_UNABLE,
	SS_SWITCH_ON_READY,
	SS_SWITCH_ON,
	SS_OPERATION_ENABLE,
	SS_FAULT,
	SS_FAULT_REACTION_ACTIVE,
	SS_MAIN_POWER_ON,
	SS_LEVEL5_ERROR
};

#define PDO_AXIS_NUM	8	

extern unsigned long int world_tick;	//Operating time since run (supports up to 49710 days of continuous operation)

const double robotGain[6] = { 100.0, 60.0, 60.0, 40.0, 70.0, 30.0 };	//Modified 20240221 Kai

// for PD control
const double robotGain_Dcontrol[6] = { 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0 };

/////////////////////////////////////////////////////////////////////////////////////////
//Log Parameter Settings
/////////////////////////////////////////////////////////////////////////////////////////

#define LOG_TIME 15	//Logs every 250[us] and can save for LOG_TIME

// {int(4byte) * 6 axes * 13 elements (command position, command velocity, current position, current velocity, target position for orbit generation calculation, target velocity for orbit generation calculation, force sensor value, current position and attitude of object, target position and attitude of object, target velocity of object, position based on target orbit, velocity based on target orbit, current external force applied to object) + status(1byte) + int(4byte) * } * number of 1s saved * sec->min * total_minutes + reserve
const unsigned long int total_log_space = (4 * 6 * 13 + 1 + 4 * 2) * FREQ * 60 * LOG_TIME + 1;

/////////////////////////////////////////////////////////////////////////////////////////
//Setting parameters for data storage
/////////////////////////////////////////////////////////////////////////////////////////

//Memory area to be allocated to store data = (Byte -> KB * KB -> MB * MB -> GB) * Allocated area (GB): If not cast, it will overflow in the middle calculation formula.
const unsigned long total_save_space = static_cast<unsigned long>(1024 * 1024 * 1024) * 1;

/////////////////////////////////////////////////////////////////
// Limit Information
/////////////////////////////////////////////////////////////////	

//-----------------Use for Denso VS068-----------------

// Maximum deviation of speed (absolute value)
// Limit for moving up to 60[deg] in 20[s] by default
const double desAcc_max_limit[6] = { 3.0, 3.0, 3.0, 3.0, 3.0, 3.0 };		// Unit [deg/s]

//Max. input speed
const double desVel_max_limit[6] = { 250.0, 250.0, 250.0, 250.0, 250.0, 500.0 };	// Unit [deg/s]

//Position error between theoretical and current values															
const double pos_error_limit[6] = { 5.0, 5.0, 5.0, 5.0, 5.0, 10.0 };

//Difference between the previous value and the position of the current value															
const double pos_diff_limit[6] = { 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 };

//20181008 (-5[deg] offset from hardware limit)
const double pos_max_limit[6] = { 165.0, 130.0, 148.0, 265.0, 114.0, 355.0 };

//20181008 (+5[deg] offset from hardware limit)
const double pos_min_limit[6] = { -165.0, -95.0, -115.0, -265.0, -114.0, -355.0 };

// How much offset to the hardware limit for each joint angle 
// (hardware limits are also implemented programmatically, but exceeding them even momentarily will break the robot)
const double hardware_limit_offset[6] = { 1.0, 1.0,  1.0,  1.0,  1.0,  1.0 };

// Limit of force applied to force sensor
const double force_torque_limit_max[6] = { 20.0, 20.0, 20.0, 3.0, 3.0, 3.0 };
const double force_torque_limit_min[6] = { -20.0, -20.0, -20.0, -3.0, -3.0, -3.0 };

//-----------------Use for YASKAWA-----------------
// Maximum input speed
// Limit for moving up to 60[deg] in 20[s] by default
const double desVel_max_limit_YASKAWA[3] = {7200.0, 1000.0, 1000.0};// Unit [deg/s]

// Position error between theoretical and current values															
const double pos_error_limit_YASKAWA[3] = {5.0, 5.0, 5.0};

// Difference between the previous value and the position of the current value															
const double pos_diff_limit_YASKAWA[3] = {5.0, 2.0, 2.0};

// Maximum value of each joint angle
const double pos_max_limit_YASKAWA[3] = {1182000.0, 6800.0, 9000.0};

// Minimum value of each joint angle
//20181008 (+5[deg] offset from hardware limit)
const double pos_min_limit_YASKAWA[3] = {-1182000.0, 0.0, 0.0};

// How much offset to hardware limit for each joint angle 
const double hardware_limit_offset_YASKAWA[3] = { 1.0, 1.0, 1.0};

#endif // !GLOBAL_H

