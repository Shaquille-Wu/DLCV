#include "test_common.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include "logging.h"

#define DLCV_MAX_PATH 1024

bool ReadFileToBuffer(const char* filename, std::vector<char> &buffer) {
	std::ifstream file(filename, std::ios::binary);
	if (file.good())
	{
		file.seekg(0, file.end);
		uint64_t fileLength = file.tellg();
		file.seekg(0, file.beg);
		buffer.resize(fileLength);
		file.read((char*)buffer.data(), fileLength);
		file.close();
	}
	return !buffer.empty();
}

void StringSplit(const std::string &s, const std::string seperator, std::vector<std::string>& result) {
	typedef std::string::size_type string_size;
	string_size i = 0;

	while (i != s.size()) {
		//找到字符串中首个不等于分隔符的字母；
		int flag = 0;
		while (i != s.size() && flag == 0) {
			flag = 1;
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[i] == seperator[x]) {
					++i;
					flag = 0;
					break;
				}
		}

		//找到又一个分隔符，将两个分隔符之间的字符串取出；
		flag = 0;
		string_size j = i;
		while (j != s.size() && flag == 0) {
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[j] == seperator[x]) {
					flag = 1;
					break;
				}
			if (flag == 0)
				++j;
		}
		if (i != j) {
			result.push_back(s.substr(i, j - i));
			i = j;
		}
	}
}

bool MatchedSuffix(char* file, std::vector<std::string> &suffixList) {
	std::string f(file);
	auto pos = f.rfind(".");
	auto suffix = f.substr(pos+1);
	for (auto s : suffixList) {
		if (s == suffix) {
			return true;
		}
	}
	return false;
}

#ifdef _WIN32
#include <Windows.h>
void RecursiveFile(const char* lpPath, std::vector<std::string>& suffixList, std::vector<std::string> &fileList) {
	char szFind[MAX_PATH];
	WIN32_FIND_DATAA FindFileData;
	strcpy_s(szFind, lpPath);
	strcat_s(szFind, "\\*.*");
	HANDLE hFind = ::FindFirstFileA(szFind, &FindFileData);
	if (INVALID_HANDLE_VALUE == hFind)
		return;
	while (true) {
		if (FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
			if (FindFileData.cFileName[0] != '.') {
				char szFile[MAX_PATH];
				strcpy_s(szFile, lpPath);
				strcat_s(szFile, "\\");
				strcat_s(szFile, (char*)(FindFileData.cFileName));
				RecursiveFile(szFile, suffixList, fileList);
			}
		}
		else {
			if (MatchedSuffix((char*)(FindFileData.cFileName), suffixList)) {
				char szFile[MAX_PATH];
				strcpy_s(szFile, lpPath);
				strcat_s(szFile, "\\");
				strcat_s(szFile, (char*)(FindFileData.cFileName));
				fileList.push_back(std::string(szFile));

			}
		}
		if (!FindNextFileA(hFind, &FindFileData)) {
			break;
		}
	}
	FindClose(hFind);
}

void RecursiveFolder(const char* lpPath, std::string& folderName, std::vector<std::string> &folderList) {
    //--TODO:
}

void RecursiveFolder(const char* lpPath, std::vector<std::string> &folderList) {
    char szFind[MAX_PATH];
    WIN32_FIND_DATAA FindFileData;
    strcpy_s(szFind, lpPath);
    strcat_s(szFind, "\\*.*");
    HANDLE hFind = ::FindFirstFileA(szFind, &FindFileData);
    if (INVALID_HANDLE_VALUE == hFind)
        return;
    while (true) {
        if (FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            if (FindFileData.cFileName[0] != '.') {
                char szFile[MAX_PATH];
                strcpy_s(szFile, lpPath);
                strcat_s(szFile, "\\");
                strcat_s(szFile, (char*)(FindFileData.cFileName));
                folderList.push_back(szFile);
            }
        }
        if (!FindNextFileA(hFind, &FindFileData)) {
            break;
        }
    }
    FindClose(hFind);
}

#else
#include <dirent.h>
#include <string.h>
void RecursiveFile(const char* lpPath, std::vector<std::string>& suffixList, std::vector<std::string> &fileList) {
    DIR* pDir=opendir(lpPath);
    if(!pDir) {
        LOG(FATAL) << lpPath << " not exist!!!";
        return;
    }

    char childdir[512] = {0};
    struct dirent* ent;
    while((ent=readdir(pDir))!=NULL)
    {
        if(ent->d_type & DT_DIR)
        {
            if(strcmp(ent->d_name,".")==0 || strcmp(ent->d_name,"..")==0)
                continue;
            sprintf(childdir,"%s/%s",lpPath,ent->d_name);
            RecursiveFile(childdir, suffixList, fileList);
        }
        else
        {
			if (MatchedSuffix(ent->d_name, suffixList)) {
                fileList.push_back(std::string(lpPath) + std::string("/") + std::string(ent->d_name));
            }
        }
    }
    closedir(pDir);
}

void RecursiveFolder(const char* lpPath, std::string& folderName, std::vector<std::string> &folderList) {
    DIR* pDir=opendir(lpPath);
    if(!pDir) {
        LOG(FATAL) << lpPath << " not exist!!!";
        return;
    }

    char childdir[512] = {0};
    struct dirent* ent;
    while((ent=readdir(pDir))!=NULL)
    {
        if(ent->d_type & DT_DIR)
        {
            if(strcmp(ent->d_name,".")==0 || strcmp(ent->d_name,"..")==0)
                continue;
			if (ent->d_name == folderName) {
                folderList.push_back(std::string(lpPath) + std::string("/") + std::string(ent->d_name));
                continue;
            }
            sprintf(childdir,"%s/%s",lpPath,ent->d_name);
            RecursiveFolder(childdir, folderName, folderList);
        }
    }
    closedir(pDir);
}

void RecursiveFolder(const char* lpPath,  std::vector<std::string> &folderList) {
    DIR* pDir = opendir(lpPath);
    if (!pDir) {
        LOG(FATAL) << lpPath << " not exist!!!";
        return;
    }

    struct dirent* ent;
    while ((ent = readdir(pDir)) != NULL)
    {
        if (ent->d_type & DT_DIR)
        {
            if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
                continue;

            folderList.push_back(std::string(lpPath) + std::string("/") + std::string(ent->d_name));
        }
    }
    closedir(pDir);
}

#endif
