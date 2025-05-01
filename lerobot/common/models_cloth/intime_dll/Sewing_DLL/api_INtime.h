#pragma once
#define API_INTIME __declspec(dllexport)

typedef struct {
	int state_A;
	int state_B;
	int state_vs; // use for transfer task mode from INtime to python
} State_Loading;

typedef struct {
	int state_A;
	int state_B;
} State_Sewing;

extern "C" API_INTIME int Py_Init_SM_A(void);
extern "C" API_INTIME void Py_Read_SM_A(double* PyCurpos);
extern "C" API_INTIME int Py_Read_Status_A(void);

extern "C" API_INTIME int Py_Init_SM_B(void);
extern "C" API_INTIME void Py_Read_SM_B(double* PyCurpos);
extern "C" API_INTIME int Py_Read_Status_B(void);

extern "C" API_INTIME int Py_Init_SM_A_Sewing(void);
extern "C" API_INTIME int Py_Init_SM_B_Sewing(void);
extern "C" API_INTIME State_Sewing Py_Read_Sewing_State(void);
extern "C" API_INTIME int Py_Init_SM_SewingFeedback(void);
extern "C" API_INTIME double Py_Read_Sewing_Speed(void);
extern "C" API_INTIME int Py_Write_Sewing_Feedback(double Rotation_Speed, int desired_stitch, bool finFlag, unsigned int sewFlag);
extern "C" API_INTIME int Py_Read_Final_Sewn_Stitches(void);

extern "C" API_INTIME int Py_Init_SM_A_Positioning(void);
extern "C" API_INTIME int Py_Init_SM_B_Positioning(void);
extern "C" API_INTIME State_Loading Py_Read_Loading_Alignment_State(double* PyCurEndPos_A, double* PyCurEndPos_B);
extern "C" API_INTIME int Py_Write_Loading_Alignment(double v_x_A, double v_y_A, double v_tz_A, double v_x_B, double v_y_B, double v_tz_B, bool finFlag);

extern "C" API_INTIME int Py_Init_SM_General_Master(void);
extern "C" API_INTIME void Py_Write_SM_General_Master(int name_task);
extern "C" API_INTIME int Py_Read_General_Task_Init_Flag(void);
extern "C" API_INTIME int Py_Read_General_Task_End_Flag_A(void);
extern "C" API_INTIME int Py_Read_General_Task_End_Flag_B(void);

extern "C" API_INTIME int Py_Init_SM_Sewing_Machine_Master(void);
extern "C" API_INTIME void Py_Write_SM_Sewing_Machine_Master(int name_task, double inputVal[10]);

extern "C" API_INTIME int Py_Init_SM_Grasp_Release_Master(void);
extern "C" API_INTIME void Py_Write_SM_Grasp_Release_Master(int task_A, int task_B, char inputVal_A[1024], char inputVal_B[1024]);

extern "C" API_INTIME int Py_Init_SM_EndEffector_Movement_Master(void);
extern "C" API_INTIME void Py_Write_SM_EndEffector_Movement_Master(int task_A, int task_B, char inputVal_A[1024], char inputVal_B[1024]);

extern "C" API_INTIME int Py_Init_SM_Fabric_Movement_Master(void);
extern "C" API_INTIME void Py_Write_SM_Fabric_Movement_Master(int task, char inputVal_A[1024], char inputVal_B[1024]);

extern "C" API_INTIME int Py_Init_SM_Sewing_Master(void);
extern "C" API_INTIME void Py_Write_SM_Sewing_Master(int task, char inputVal_A[1024], char inputVal_B[1024]);

extern "C" API_INTIME int Py_Init_SM_Positioning_VS_Master(void);
extern "C" API_INTIME void Py_Write_SM_Positioning_VS_Master(int task, char inputVal_A[1024], char inputVal_B[1024]);

extern "C" API_INTIME int Py_Init_SM_Teleoperate_Master(void);
extern "C" API_INTIME void Py_Write_SM_Teleoperate_Master(int task, char inputVal_A[1024], char inputVal_B[1024]);

// api_tele
extern "C" API_INTIME int Py_Init_SM_Teleoperate_A(void);
extern "C" API_INTIME int Py_Init_SM_Teleoperate_B(void);
extern "C" API_INTIME int Py_Init_SM_Teleoperation(void);
extern "C" API_INTIME void Py_Teleoperation_w2i(double DesFT_A[6], double DesFT_B[6]);
