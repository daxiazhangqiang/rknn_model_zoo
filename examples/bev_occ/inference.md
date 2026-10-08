# 感知模型推理
根据当前打开的rknn_model_zoo的项目中examples目录下的示例，完成examples/bev_occ项目的CPP和python推理代码
## 背景
1. 当前的rknn_model_zoo的项目是瑞芯微github仓库拉下来的部署仓库，阅读下当前的代码仓库，examples是其他模型推理示例
2. examples/multi_task_dacheng是我之前实现的一个示例
3. examples/bev_occ是感知模型，有三个主要的输出，静态障碍物语义分割，动态障碍物语义分割，BEVOCC.
4. examples/bev_occ/model/bevocc_96_lossweight2.rknn是我自行转换的rknn，转换脚本用的是examples/bev_occ/python/quantize_bev_occ.py
5. examples/bev_occ/python/infer_onnx_bevocc_realtime.py是onnx之前写的推理脚本，你可以作为参考，尤其是图像预处理部分，减去的均值和方差，以及模型输入是RGB格式
6. examples/bev_occ/imgs里有一张你可以使用的测试图像，尺寸是HxW=530x730
## 任务要求
1. 跟examples其他模型类型，实现python和CPP的推理代码，并且支持可视化。也就是仿照examples/bev_occ/python/infer_onnx_bevocc_realtime.py中onnx推理后结果的可视化逻辑，把三个head的模型结果叠加到输入图像上，要求输入是单张图像，比如test.jpg. 输出是三张图像， test_static.jpg,test_dynamic.jpg,test_bevocc.jpg,
其中bevocc输出暂时不需要叠加，只显示原始输出皆可，也就是64x64的灰度图做颜色的可视化即可，主要是因为内参参暂时不准。
2. python/CPP推理脚本默认的功能为
python inference.py model.rknn {img or img_folder} [{out_img or out_img_folder}], 比如： 
infer_program model.rknn test.img/test_img_folder [bevocc_out]
CPP脚本也是如此，追求推理功能输入的简洁性， 另外该模型需要相机内外参，当前onnx中已经固化进去了相机内外参，暂时不需要用户输入。
具体功能为输入就是图像时，输出是输入图像名称为名的文件夹，里面是3张可视化图像。输出目录允许用户指定，如果用户没有指定就默认用bevocc_out作为输出文件夹，下一级目录则是输入图像文件夹里每张图像为名称的文件夹。如果输入是图像文件夹，则遍历其中所有图像进行推理。这种使用方法方面可以直接参考examples/multi_task_dacheng中使用推理程序的方法。
3. CPP的代码中加一个宏用于控制是否进行可视化，因为正式发布时是不需要可视化的，另外CPP测试时记得完整编译测试，当前环境中有编译工具连，也有NPU.
4. 功能开发完成，用examples/bev_occ/imgs进行测试

## 任务结果

## 注意事项
1. 过程中出现的问题自行解决，不用问我，我给你开了终端执行指令和编辑文件的权限，只需要最后把你过程中的关键选择汇总给我即可。