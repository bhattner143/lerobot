#ifndef UNIVERSALSM_EXE_H
#define UNIVERSALSM_EXE_H

#include <ntx.h>

namespace USM
{
	namespace EXE {
		class Access
		{
		public:
			Access(void);
			~Access(void);

			int Read(unsigned char* readAddr, unsigned long startOffset, unsigned long readSize);
			int Write(unsigned char* writeAddr, unsigned long startOffset, unsigned long writeSize);

			void SetHandle(NTXHANDLE tmp_handle);
			void SetHandleEx(NTXHANDLE tmp_handle, DWORD flag);

		private:
			NTXHANDLE handle;
		
			bool handleFlag;
			BYTE* getSM;
		};
	}
}


#endif //UNIVERSALSM_Exe_H