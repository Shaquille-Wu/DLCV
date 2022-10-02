#ifndef __DLCV_TEST_COMMON_H__
#define __DLCV_TEST_COMMON_H__

#include <vector>
#include <string>

bool ReadFileToBuffer(const char* filename, std::vector<char> &buffer);
void StringSplit(const std::string &s, const std::string seperator, std::vector<std::string>& result);
void RecursiveFile(const char* lpPath, std::vector<std::string>& suffixList, std::vector<std::string> &fileList);
void RecursiveFolder(const char* lpPath, std::string& folderName, std::vector<std::string> &folderList);
void RecursiveFolder(const char* lpPath, std::vector<std::string> &folderList);

#endif //__DLCV_TEST_COMMON_H__
