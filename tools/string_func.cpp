/*
 * Copyright (C) OrionStar Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file string_func.cpp
 * @brief the implementation for some basic and common operation of string
 * @author wuxiao@ainirobot.com
 * @date 2020-03-12
 */

#include "string_func.h"
#include <dirent.h>
#include <sys/stat.h>
#include <string.h>
#include <algorithm>

std::string trim_string(const std::string& src)
{
    int    str_len     = src.length();
    int    start_pos   = 0;
    int    end_pos     = 0;
    int    i           = 0;
    char*  trim_str    = new char[str_len + 1];
    memset(trim_str, 0, str_len + 1);
    for(i = 0 ; i < str_len ; i ++)
    {
        if(' ' != (src.c_str())[i] && '\r' != (src.c_str())[i] && '\n' != (src.c_str())[i])
        {
            start_pos = i;
            break;
        }
    }
    for(i = str_len - 1 ; i >= 0 ; i --)
    {
        if(' ' != (src.c_str())[i] && '\r' != (src.c_str())[i] && '\n' != (src.c_str())[i])
        {
            end_pos = i;
            break;
        }
    }

    int trim_len = end_pos - start_pos + 1;
    for(i = 0 ; i < trim_len ; i ++)
        trim_str[i] = (src.c_str())[i + start_pos];

    std::string  result = std::string(trim_str);
    delete[] trim_str;

    return result;
}

std::string               get_file_ext(const std::string& src)
{
    int    len        = src.length();
    char*  ext        = new char[len + 1];
    int    start_pos  = 0;
    int    i          = 0;

    for(i = len - 1 ; i >= 0 ; i --)
    {
        if('.' == src.c_str()[i])
        {
            start_pos = i;
            break;
        }
    }

    memset(ext, 0, len + 1);
    for(i = start_pos ; i < len ; i ++)
    {
        ext[i - start_pos] = src.c_str()[i];
    }

    std::string   result = ext;

    delete[] ext;
    return result;
}

#define MAX_PATH_LEN 512
std::vector<std::string>  travel_image_dir(const std::string&                src_image_dir, 
                                           const std::vector<std::string>&   ext_name_list,
                                           const std::string&                sub_dir)
{
    DIR*                         d                  = NULL;
    struct dirent*               dp                 = NULL;
    struct stat                  st;    
    char                         p[MAX_PATH_LEN]    = {0};
    std::vector<std::string>     image_file_list;
    int                          i                  = 0;
    int                          ext_name_list_cnt  = (int)(ext_name_list.size());
    std::string                  cur_sub_dir        = "";

    if(stat(src_image_dir.c_str(), &st) < 0 || !S_ISDIR(st.st_mode)) {
        printf("invalid path: %s\n", src_image_dir.c_str());
        return image_file_list;
    }

    if(!(d = opendir(src_image_dir.c_str()))) {
        printf("opendir[%s] error: %m\n", src_image_dir.c_str());
        return image_file_list;
    }

    while((dp = readdir(d)) != NULL) 
    {
        if((!strncmp(dp->d_name, ".", 1)) || (!strncmp(dp->d_name, "..", 2)))
            continue;

        snprintf(p, sizeof(p) - 1, "%s/%s", src_image_dir.c_str(), dp->d_name);
        stat(p, &st);
        if(!S_ISDIR(st.st_mode)) 
        {
            std::string    cur_file      = dp->d_name;
            int            str_len       = cur_file.length();
            bool           found         = false;
            std::string    cur_file_ext  = get_file_ext(cur_file);

            for(i = 0 ; i < ext_name_list_cnt ; i ++)
            {
                const std::string&  cur_ext_name = ext_name_list.at(i);
                if(0 == strcasecmp(cur_ext_name.c_str(), cur_file_ext.c_str()))
                {
                    found = true;
                    break;
                }
            }
            if(true == found)
            {
                if("" != sub_dir)
                    cur_file = sub_dir + std::string("/") + cur_file;
                image_file_list.push_back(cur_file);
            }
        } 
        else
        {
            if("" == sub_dir)
                cur_sub_dir = dp->d_name;
            else
                cur_sub_dir = sub_dir + "/" + dp->d_name;
            std::vector<std::string> sub_image_file_list = travel_image_dir(std::string(p), ext_name_list, cur_sub_dir);
            image_file_list.insert(image_file_list.end(),sub_image_file_list.begin(),sub_image_file_list.end());
        }
    }
    closedir(d);

    std::sort(image_file_list.begin(), image_file_list.end());

    return image_file_list;
}