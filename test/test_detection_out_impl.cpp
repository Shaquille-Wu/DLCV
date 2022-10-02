#include "gtest/gtest.h"
#include "./detection_out/detection_out_impl.h"

static const int                     kImageWidth                              = 320;
static const int                     kImageHeight                             = 320;
static const int                     kPriorBoxLayerCount                      = 5;
static const vision::PriorBoxParam   kPriorBoxLayerParam[kPriorBoxLayerCount] = {   
    {    //neck/priorbox_0
         {                     //min_size
            16.0f,
            20.1587371826f,
            25.3984165192f,
         },            
         {                     //max_size
         },                 
         {                     //aspect_ratio
            1.0f,
            0.5f
         },                    
         false,                //flip
         false,                //clip
         {                     //variance
            0.10000000149f, 
            0.10000000149f, 
            0.20000000298f, 
            0.20000000298f 
         }, 
         8.0f,                 //step_size
         0.5f,                 //offset
         40                    //layer_size
    },
    {    //neck/priorbox_1
        {                      //min_size
            32.0f,
            40.3174743652f,
            50.7968330383f, 
        },   
        {                      //max_size

        },   
        {                      //aspect_ratio
            1.0f,
            0.5f
        },  
        false,                 //flip
        false,                 //clip
        {                      //variance
            0.10000000149f, 
            0.10000000149f, 
            0.20000000298f, 
            0.20000000298f 
        }, 
        16.0f,                 //step_size
        0.5f,                  //offset
        20                     //layer_size
    },

    {   //neck/priorbox_2
        {                      //min_size
            64.0f,
            80.6349487305f,
            101.593666077f
        },   
        {                      //max_size
        },   
        {                      //aspect_ratio
            1.0f,
            0.5f
        },             
        false,                 //flip
        false,                 //clip
        {                      //variance
            0.10000000149f, 
            0.10000000149f, 
            0.20000000298f, 
            0.20000000298f 
        }, 
        32.0f,                 //step_size
        0.5f,                  //offset   
        10                     //layer_size
    },

    {   //neck/priorbox_3
        {                      //min_size       
            128.0f,
            161.269897461f,
            203.187332153f
        },   
        {                      //max_size

        },   
        {                      //aspect_ratio
            1.0f,
            0.5f 
        },             
        false,                 //flip
        false,                 //clip
        {                      //variance
            0.10000000149f, 
            0.10000000149f, 
            0.20000000298f, 
            0.20000000298f 
        }, 
        64.0f,                 //step_size
        0.5f,                  //offset
        5                      //layer_size
    },

    {   //neck/priorbox_4
        {                      //min_size 
            256.0f,
            322.539794922f,
            406.374664307f
        },  
        {                     //max_size

        },           
        {                     //aspect_ratio
            1.0f,
            0.5f,
        },                    
        false,                //flip
        false,                //clip
        {                     //variance
            0.10000000149f, 
            0.10000000149f, 
            0.20000000298f, 
            0.20000000298f 
        }, 
        128.0f,               //step_size
        0.5f,                 //offset
        3                     //layer_size
    },           
};

static const vision::DetectionOutParam   kDetectionOutParam = {
    1,                              //class_count
    -2,                             //background_label_id_
    0.05f,                          //confidence threshold for result
    0.4f,                           //nms threshold for result
    100,                            //top_k_
    vision::CODE_TYPE_CENTER_SIZE,  //bbox_code_type_
};

static const int             kQuantizedZero          = 114;
static const float           kQuantizedStep          = 1.077614;
static const std::string     kConfTensor             = "neck/conf_sigmoid";
static const std::string     kLocTensor              = "neck/loc_concat";
static const std::string     kResultTensor           = "neck/detection_out";

static int  ComparePriorBoxParam(const vision::PriorBoxParam& prior_box_param_a, const vision::PriorBoxParam& prior_box_param_b)
{
    int j = 0;
    if(prior_box_param_a.min_size_.size() != prior_box_param_b.min_size_.size())            return -1;
    for(j = 0 ; j < (int)(prior_box_param_a.min_size_.size()) ; j ++)
    {
        if(prior_box_param_a.min_size_[j] != prior_box_param_b.min_size_[j])                return -1;    
    }
    if(prior_box_param_a.max_size_.size() != prior_box_param_b.max_size_.size())            return -1;
    for(j = 0 ; j < (int)(prior_box_param_a.max_size_.size()) ; j ++)
    {
        if(prior_box_param_a.max_size_[j] != prior_box_param_b.max_size_[j])                return -1;    
    }  
    if(prior_box_param_a.aspect_ratio_.size() != prior_box_param_b.aspect_ratio_.size())    return -1;
    for(j = 0 ; j < (int)(prior_box_param_a.aspect_ratio_.size()) ; j ++)
    {
        if(prior_box_param_a.aspect_ratio_[j] != prior_box_param_b.aspect_ratio_[j])        return -1;    
    }     
    if(prior_box_param_a.flip_ != prior_box_param_b.flip_)                                  return -1;     
    if(prior_box_param_a.clip_ != prior_box_param_b.clip_)                                  return -1;   

    if(prior_box_param_a.variance_.size() != prior_box_param_b.variance_.size())            return -1;
    for(j = 0 ; j < (int)(prior_box_param_a.variance_.size()) ; j ++)
    {
        if(prior_box_param_a.variance_[j] != prior_box_param_b.variance_[j])                return -1;    
    }   

    if(prior_box_param_a.step_size_ != prior_box_param_b.step_size_)                        return -1;     
    if(prior_box_param_a.offset_ != prior_box_param_b.offset_)                              return -1; 
    if(prior_box_param_a.layer_size_ != prior_box_param_b.layer_size_)                      return -1; 

    return 0;
}

static int  CompareDetectionOutParam(const vision::DetectionOutParam& detection_out_param_a, 
                                     const vision::DetectionOutParam& detection_out_param_b)
{
    if(detection_out_param_a.class_count_ != detection_out_param_b.class_count_)                    return -1;
    if(detection_out_param_a.background_label_id_ != detection_out_param_b.background_label_id_)    return -1;
    if(detection_out_param_a.conf_threshold_ != detection_out_param_b.conf_threshold_)              return -1;
    if(detection_out_param_a.nms_threshold_ != detection_out_param_b.nms_threshold_)                return -1;
    if(detection_out_param_a.top_k_ != detection_out_param_b.top_k_)                                return -1;
    if(detection_out_param_a.bbox_code_type_ != detection_out_param_b.bbox_code_type_)              return -1;

    return 0;
}

static int  CompareDetectOutAllParam(const vision::DetectionOutImpl&  detection_out_impl_a, 
                                     const vision::DetectionOutImpl&  detection_out_impl_b)
{
    int  res = 0, i = 0, j = 0;
    if(detection_out_impl_a.GetImageWidth()  != detection_out_impl_b.GetImageWidth())                      return -1;
    if(detection_out_impl_a.GetImageHeight() != detection_out_impl_b.GetImageHeight())                    return -1;
    if(detection_out_impl_a.GetPriorBoxParam().size() != detection_out_impl_b.GetPriorBoxParam().size())  return -1;
    for(i = 0 ; i < (int)(detection_out_impl_a.GetPriorBoxParam().size()) ; i ++)
    {
        res = ComparePriorBoxParam(detection_out_impl_a.GetPriorBoxParam()[i], detection_out_impl_b.GetPriorBoxParam()[i]);
        if(0 != res)    return -1;
    }
    res = CompareDetectionOutParam(detection_out_impl_a.GetDetectionOutParam(),
                                   detection_out_impl_a.GetDetectionOutParam());
    if(0 != res)    return -1;  

    if(detection_out_impl_a.GetQuantizationParameter().zero_ != detection_out_impl_b.GetQuantizationParameter().zero_)   return -1;
    if(detection_out_impl_a.GetQuantizationParameter().step_ != detection_out_impl_b.GetQuantizationParameter().step_)   return -1;
    if(detection_out_impl_a.GetOutputConfTensor() != detection_out_impl_b.GetOutputConfTensor())         return -1;
    if(detection_out_impl_a.GetOutputLocTensor() != detection_out_impl_b.GetOutputLocTensor())           return -1;
    if(detection_out_impl_a.GetOutputResultTensor() != detection_out_impl_b.GetOutputResultTensor())     return -1;

    if(detection_out_impl_a.GetImageWidth() != kImageWidth)                      return -1;
    if(detection_out_impl_a.GetImageHeight() != kImageHeight)                    return -1;
    if(detection_out_impl_a.GetPriorBoxParam().size() != kPriorBoxLayerCount)    return -1;
    for(i = 0 ; i < kPriorBoxLayerCount ; i ++)
    {
        res = ComparePriorBoxParam(detection_out_impl_a.GetPriorBoxParam()[i], kPriorBoxLayerParam[i]);
        if(0 != res)    return -1;
    }
    res = CompareDetectionOutParam(detection_out_impl_a.GetDetectionOutParam(),
                                   kDetectionOutParam);
    if(0 != res)    return -1;  

    if(detection_out_impl_a.GetQuantizationParameter().zero_ != kQuantizedZero)   return -1;
    if(detection_out_impl_a.GetQuantizationParameter().step_ != kQuantizedStep)   return -1;
    if(detection_out_impl_a.GetOutputConfTensor() != kConfTensor)                 return -1;
    if(detection_out_impl_a.GetOutputLocTensor() != kLocTensor)                   return -1;

    return 0;
}

TEST(DetectionOut, Default_Constructor) 
{
    vision::DetectionOutImpl    det_out_impl;
    ASSERT_EQ(det_out_impl.GetResultData(), nullptr);
    ASSERT_EQ(det_out_impl.GetResultItemCount(), 0);
    ASSERT_EQ(det_out_impl.GetResultDataCount(), 0);
    ASSERT_EQ(det_out_impl.GetPriorBoxCount(), 0);
    ASSERT_EQ(det_out_impl.GetImageWidth(),  0);
    ASSERT_EQ(det_out_impl.GetImageHeight(), 0);    
    ASSERT_EQ(det_out_impl.GetPriorBoxParam().size(), 0);
    ASSERT_EQ(det_out_impl.GetDetectionOutParam().class_count_, 1);
    ASSERT_EQ(det_out_impl.GetDetectionOutParam().background_label_id_, 0);
    ASSERT_EQ(det_out_impl.GetDetectionOutParam().conf_threshold_, 0.0f);
    ASSERT_EQ(det_out_impl.GetDetectionOutParam().nms_threshold_, 0.0f);
    ASSERT_EQ(det_out_impl.GetDetectionOutParam().top_k_, 0);
    ASSERT_EQ(det_out_impl.GetDetectionOutParam().bbox_code_type_, vision::PRIOR_BOX_CODE_TYPE::CODE_TYPE_CENTER_SIZE);
    ASSERT_EQ(det_out_impl.GetQuantizationParameter().zero_, 0);
    ASSERT_EQ(det_out_impl.GetQuantizationParameter().step_, 0);
    ASSERT_EQ(det_out_impl.GetOutputConfTensor(), "");
    ASSERT_EQ(det_out_impl.GetOutputLocTensor(), "");
}

TEST(DetectionOut, Parameter_Constructor) 
{
    int    i = 0;
    std::vector<vision::PriorBoxParam>   vec_prior_box;
    for(i = 0 ; i < kPriorBoxLayerCount ; i ++)
        vec_prior_box.push_back(kPriorBoxLayerParam[i]);

    vision::DetectionOutImpl     det_out_impl(kImageWidth, 
                                              kImageHeight, 
                                              vec_prior_box, 
                                              kDetectionOutParam,
                                              vision::QuantizationParam(kQuantizedZero, kQuantizedStep),
                                              kConfTensor,
                                              kLocTensor);

    ASSERT_EQ(det_out_impl.GetImageWidth(),  kImageWidth);
    ASSERT_EQ(det_out_impl.GetImageHeight(), kImageHeight);    
    ASSERT_EQ(det_out_impl.GetPriorBoxParam().size(), vec_prior_box.size());

    for(i = 0 ; i < (int)(det_out_impl.GetPriorBoxParam().size()) ; i ++)
    {
        ASSERT_EQ(0, ComparePriorBoxParam(det_out_impl.GetPriorBoxParam()[i], kPriorBoxLayerParam[i]));
    }
    ASSERT_EQ(0, (det_out_impl.GetQuantizationParameter().zero_ == kQuantizedZero));
    ASSERT_EQ(0, (det_out_impl.GetQuantizationParameter().step_ == kQuantizedStep));

    ASSERT_TRUE(det_out_impl.GetOutputConfTensor() == kConfTensor);
    ASSERT_TRUE(det_out_impl.GetOutputLocTensor() == kLocTensor);

    ASSERT_EQ(det_out_impl.GetResultData(), nullptr);
    ASSERT_EQ(det_out_impl.GetResultItemCount(), 0);
    ASSERT_EQ(det_out_impl.GetResultDataCount(), 0);
    ASSERT_EQ(det_out_impl.GetPriorBoxCount(), 0);
}

TEST(DetectionOut, Copy_Constructor) 
{
    int    i = 0;
    std::vector<vision::PriorBoxParam>   vec_prior_box;
    for(i = 0 ; i < kPriorBoxLayerCount ; i ++)
        vec_prior_box.push_back(kPriorBoxLayerParam[i]);

    vision::DetectionOutImpl     det_out_impl_a(kImageWidth, 
                                                kImageHeight, 
                                                vec_prior_box, 
                                                kDetectionOutParam,
                                                vision::QuantizationParam(kQuantizedZero, kQuantizedStep),
                                                kConfTensor,
                                                kLocTensor);

    vision::DetectionOutImpl     det_out_impl_b(det_out_impl_a);

    ASSERT_EQ(0, CompareDetectOutAllParam(det_out_impl_a, det_out_impl_b));

    ASSERT_EQ(det_out_impl_a.GetResultData(), det_out_impl_b.GetResultData());
    ASSERT_EQ(det_out_impl_a.GetResultItemCount(), det_out_impl_b.GetResultItemCount());
    ASSERT_EQ(det_out_impl_a.GetResultDataCount(), det_out_impl_b.GetResultDataCount());
    ASSERT_EQ(det_out_impl_a.GetPriorBoxCount(), det_out_impl_b.GetPriorBoxCount());

    ASSERT_EQ(det_out_impl_a.GetResultData(), nullptr);
    ASSERT_EQ(det_out_impl_a.GetResultItemCount(), 0);
    ASSERT_EQ(det_out_impl_a.GetResultDataCount(), 0);
    ASSERT_EQ(det_out_impl_a.GetPriorBoxCount(), 0);    
}

TEST(DetectionOut, Assign_Operator) 
{
    int    i = 0;
    std::vector<vision::PriorBoxParam>   vec_prior_box;
    for(i = 0 ; i < kPriorBoxLayerCount ; i ++)
        vec_prior_box.push_back(kPriorBoxLayerParam[i]);
    vision::DetectionOutImpl     det_out_impl_a(kImageWidth, 
                                                kImageHeight, 
                                                vec_prior_box, 
                                                kDetectionOutParam,
                                                vision::QuantizationParam(kQuantizedZero, kQuantizedStep),
                                                kConfTensor,
                                                kLocTensor);
    vision::DetectionOutImpl     det_out_impl_b;
    det_out_impl_b = det_out_impl_a;
    
    ASSERT_EQ(0, CompareDetectOutAllParam(det_out_impl_a, det_out_impl_b));

    ASSERT_EQ(det_out_impl_a.GetResultData(), det_out_impl_b.GetResultData());
    ASSERT_EQ(det_out_impl_a.GetResultItemCount(), det_out_impl_b.GetResultItemCount());
    ASSERT_EQ(det_out_impl_a.GetResultDataCount(), det_out_impl_b.GetResultDataCount());
    ASSERT_EQ(det_out_impl_a.GetPriorBoxCount(), det_out_impl_b.GetPriorBoxCount());

    ASSERT_EQ(det_out_impl_a.GetResultData(), nullptr);
    ASSERT_EQ(det_out_impl_a.GetResultItemCount(), 0);
    ASSERT_EQ(det_out_impl_a.GetResultDataCount(), 0);
    ASSERT_EQ(det_out_impl_a.GetPriorBoxCount(), 0);   
}

TEST(DetectionOut, Set_Get) 
{
    std::vector<vision::PriorBoxParam>   vec_prior_box;
    for(i = 0 ; i < kPriorBoxLayerCount ; i ++)
        vec_prior_box.push_back(kPriorBoxLayerParam[i]);

    vision::DetectionOutImpl    det_out_impl;
    det_out_impl.SetImageWidth(kImageWidth);
    det_out_impl.SetImageHeight(kImageHeight);
    det_out_impl.SetPriorBoxParam(vec_prior_box);
    det_out_impl.SetDetectionOutParam(kDetectionOutParam);
    det_out_impl.SetQuantizationParameter(orion::QuantizationParam(kQuantizedZero, kQuantizedStep));
    det_out_impl.SetOutputConfTensor(kConfTensor);
    det_out_impl.SetOutputLocTensor(kLocTensor);

    ASSERT_EQ(det_out_impl.GetImageWidth(),  kImageWidth);
    ASSERT_EQ(det_out_impl.GetImageHeight(), kImageHeight);    
    ASSERT_EQ(det_out_impl.GetPriorBoxParam().size(), vec_prior_box.size());

    for(i = 0 ; i < (int)(det_out_impl.GetPriorBoxParam().size()) ; i ++)
    {
        ASSERT_EQ(0, ComparePriorBoxParam(det_out_impl.GetPriorBoxParam()[i], kPriorBoxLayerParam[i]));
    }
    ASSERT_EQ(0, (det_out_impl.GetQuantizationParameter().zero_ == kQuantizedZero));
    ASSERT_EQ(0, (det_out_impl.GetQuantizationParameter().step_ == kQuantizedStep));

    ASSERT_TRUE(det_out_impl.GetOutputConfTensor() == kConfTensor);
    ASSERT_TRUE(det_out_impl.GetOutputLocTensor()  == kLocTensor);

    ASSERT_EQ(det_out_impl.GetResultData(), nullptr);
    ASSERT_EQ(det_out_impl.GetResultItemCount(), 0);
    ASSERT_EQ(det_out_impl.GetResultDataCount(), 0);
    ASSERT_EQ(det_out_impl.GetPriorBoxCount(), 0);    
}

TEST(DetectionOut, Setup) 
{
    std::vector<vision::PriorBoxParam>   vec_prior_box;
    for(i = 0 ; i < kPriorBoxLayerCount ; i ++)
        vec_prior_box.push_back(kPriorBoxLayerParam[i]);
    vision::DetectionOutImpl     det_out_impl(kImageWidth, 
                                              kImageHeight, 
                                              vec_prior_box, 
                                              kDetectionOutParam,
                                              vision::QuantizationParam(kQuantizedZero, kQuantizedStep),
                                              kConfTensor,
                                              kLocTensor);

    ASSERT_EQ(0, det_out_impl.Setup());

    ASSERT_EQ(12804, det_out_impl.GetPriorBoxCount());
    ASSERT_TRUE(nullptr != det_out_impl.GetResultData());
    ASSERT_EQ(det_out_impl.GetResultItemCount(), 0);
    ASSERT_EQ(det_out_impl.GetResultDataCount(), 0);
}
