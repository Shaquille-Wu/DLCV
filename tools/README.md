## preprocess_image  
一个可执行程序  
非量化模型转量化模型之前需要对原始图像进行预处理，由3通道8bit的图像转成32位float的raw文件  
命令格式：./preprocess_image -jjson_file -ssrc_image -ddst_image  
json_file: 和推理引擎用到的.json文件的内容一致，要求是一个非量化模型用到的.json文件。  
           考虑到推理引擎用到的.json文件是在已经拥有模型的情况下填写的，在模型转换阶段还不能完整提供与推理引擎匹配的.json文件。  
		   如果不能完整提供.json文件，则这个.json文件中要完整包含以下2个内容：  
           1)"preprocess"的全部  
		   2)"inference"的"inputs"的内容    
           缺失以上内容，将无法进行正常的预处理  
src_image: 原始3通道8bit图像文件，比如test.jpg  
dst_image: 预处理之后的3通道/单通道32位float的raw文件，如果填空，那么将取src_image的同名文件，只是要将扩展名改为".raw"  
命令示例:  
```
./preprocess_image -jface_model.json -s./src_image.jpg -d./dst_image.raw 
```

