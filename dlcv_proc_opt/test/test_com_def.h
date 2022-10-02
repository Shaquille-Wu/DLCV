#ifndef __TEST_COM_DEF_H__
#define __TEST_COM_DEF_H__

#ifdef __cplusplus
extern "C"{
#endif

void                GenerateFloatData(float* dst, int data_cnt);

void                GenerateUCData(unsigned char* dst, int data_cnt);

unsigned long long  GetOrionTimeUs();

#ifdef __cplusplus
}
#endif

#endif