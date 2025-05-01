#ifndef PDO_INFO_FTSENSOR
#define PDO_INFO_FTSENSOR

#pragma pack(push, 1)	//構造体のパディング防止

/////////////////////////////////////////////////////////////////
//PDOの書き込み用の先頭アドレス
/////////////////////////////////////////////////////////////////

#define AXIA_FTSENSOR_WRITE_CONTROL_1_ADDR	0
#define AXIA_FTSENSOR_WRITE_CONTROL_2_ADDR	4


/////////////////////////////////////////////////////////////////
//PDOの読み取り用の先頭アドレス
/////////////////////////////////////////////////////////////////

#define AXIA_FTSENSOR_READ_FX_ADDR	0
#define AXIA_FTSENSOR_READ_FY_ADDR	4
#define AXIA_FTSENSOR_READ_FZ_ADDR	8
#define AXIA_FTSENSOR_READ_TX_ADDR	12
#define AXIA_FTSENSOR_READ_TY_ADDR	16
#define AXIA_FTSENSOR_READ_TZ_ADDR	20
#define AXIA_FTSENSOR_READ_STATUSCODE_ADDR	24
#define AXIA_FTSENSOR_READ_SAMPLECOUNTER_ADDR	28


/////////////////////////////////////////////////////////////////
//PDOの書き込み用の構造体
/////////////////////////////////////////////////////////////////

//PDOの書き込み用構造体
typedef struct {
	unsigned long FT_ControlCodes_1;
	unsigned long FT_ControlCodes_2;
} PDO_FT_W;


/////////////////////////////////////////////////////////////////
//PDOの読み込み用の構造体
/////////////////////////////////////////////////////////////////

//PDOの読み取り用の構造体
typedef struct {
	int Fx;
	int Fy;
	int Fz;
	int Tx;
	int Ty;
	int Tz;
	unsigned long StatusCode;
	unsigned long SampleCounter;
} PDO_FT_R;

#pragma pack(pop)

#endif	// PDO_INFO_FTSENSOR
