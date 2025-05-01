#include <ntx.h>
#include <iostream>
#include "UniversalSM_Exe.h"

using namespace std;

namespace USM
{
	namespace EXE
	{
		Access::Access(void)
		{
			this->handleFlag = 0;
		}

		Access::~Access(void)
		{
			this->handleFlag = 0;
		}

		int Access::Read(unsigned char* readAddr, unsigned long startOffset, unsigned long readSize)
		{
			if (this->handleFlag == 1) {
				for (unsigned long i = 0; i < readSize; i++) {
					readAddr[i] = getSM[i + startOffset];
				}

				return 0;
			}

			return -1;
		}

		int Access::Write(unsigned char* writeAddr, unsigned long startOffset, unsigned long writeSize)
		{
			if (this->handleFlag == 1) {

				for (unsigned long i = 0; i < writeSize; i++) {
					getSM[i + startOffset] = writeAddr[i];
				}

				return 0;
			}

			return -1;
		}

		void Access::SetHandle(NTXHANDLE tmp_handle)
		{
			this->handle = tmp_handle;

			this->getSM = (BYTE*)ntxMapRtSharedMemory(this->handle);

			if (this->getSM == NULL) {
				cout << "Shared Memory Map Error : " << ntxGetLastRtError() << endl;
				return ;
			}
			this->handleFlag = 1;

		}

		void Access::SetHandleEx(NTXHANDLE tmp_handle, DWORD flag) 
		{
			this->handle = tmp_handle;

			this->getSM = (BYTE*)ntxMapRtSharedMemoryEx(this->handle, flag);

			if (this->getSM == NULL) {
				cout << "Shared Memory Map Error : " << ntxGetLastRtError() << endl;
				return;
			}
			this->handleFlag = 1;
		}

	}
}
