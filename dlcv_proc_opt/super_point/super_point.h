#ifndef __SUPER_POINT_H__
#define __SUPER_POINT_H__

int  super_point_extract_flag_implement(float const* prob, int channel, int* pt_flag, float threshold);

void normalize_super_point_feature_implement(float const*    feature_map, 
                                             int             feature_width, 
                                             int             feature_height, 
                                             int             feature_channel, 
                                             float*          normalize_feature_map);

float extract_super_point_feature_implement(float const*      feature_00, 
                                            float const*      feature_01, 
                                            float const*      feature_10, 
                                            float const*      feature_11, 
                                            float             u,
                                            float             v,
                                            int               feature_channel,
                                            float*            result);

void super_point_normalize_feature_in_place_implement(float* result, int feature_channel, float sum_rsqrt);

void super_point_nms_map_8x_implement(unsigned short int const*  pt_list, int pt_cnt, unsigned char* nms_mask_map, int map_w);

#endif