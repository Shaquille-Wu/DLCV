#include "test_com_def.h"
#include <time.h>
#include <stdlib.h>
#ifndef _MSC_VER
#include <sys/time.h>
#else
#include <Windows.h>
#endif

static int g_iRandInit = 0;

void GenerateFloatData(float* dst, int data_cnt)
{
    int i = 0 ;
    if(0 == g_iRandInit)
    {
        srand(time(0));
        g_iRandInit = 1;
    }

    for(i = 0 ; i < data_cnt ; i ++)
        dst[i] = (float)(rand()%256);
}

void GenerateUCData(unsigned char* dst, int data_cnt)
{
    int i = 0 ;
    if(0 == g_iRandInit)
    {
        srand(time(0));
        g_iRandInit = 1;
    }

    for(i = 0 ; i < data_cnt ; i ++)
        dst[i] = (unsigned char)(rand()%256);
}

unsigned long long          GetOrionTimeUs()
{
    unsigned long long time = 0;
#ifndef _MSC_VER
    struct  timeval tv_cur = { 0, 0 } ;
    gettimeofday(&tv_cur,0);
    time = (unsigned long long)(tv_cur.tv_sec) * 1000000 + tv_cur.tv_usec;
#else
    LARGE_INTEGER now, freq;
    QueryPerformanceCounter(&now);
    QueryPerformanceFrequency(&freq);
    unsigned long long sec  = now.QuadPart / freq.QuadPart;
    unsigned long long usec = (now.QuadPart % freq.QuadPart) * 1000000 / freq.QuadPart;
    time = sec * 1000000 + usec;
#endif

    return time;
}