## The purpose of this repo
Here are my tries to reduce, quantize and optimizing a unet network for embedding purpose. The idea behind such a model is to filter which picture should be sent on earth and which picture should be deleted, when doing earth observation.

Through various opportunities, I figured out the cost in terms of link budget and ground station, for communicating between orbit and ground.

## The goal
Make the inference of the model as lightweight as possible through differents steps.

## To-do list
- [x] Having a model that works pretty well
- [x] Optimizing the model
- [x] Quantize the model
- [ ] Using Rust for inference

### One last thing
Here are the acronyms used in this repo:
- PTQ: Post-training static quantization
- QAT: Quantization-aware training
- FS: From scratch
- FT: Fine-tuning (with pre-trained weights)
- FP32: full precision 32-bit floating point
- INT8: 8-bit integer quantization
- ?M or ?k: model size in number of parameters (e.g. 1M = 1e6 parameters, 400k = 400e3 parameters)

## 1 - Having a model that works pretty well
![image](inference_result_31M/test_1.png)

## 2 - Optimizing the model
For the optimization part, I tried to remove some layers in the U-Net. This leads to 4 models, summarized bellow:

|DownSample|Bottleneck|Upsample|BatchNorm|Model size (in parameters)|
|----------|----------|--------|---------|----------|
|4|1|4|No|31M|
|3|1|3|Yes|7M|
|2|1|2|Yes|1M|
|1|1|1|Yes|400k|

And the results are the following:

| |Train|Test 1|Test 2|
|-|-----|------|------|
|31M|![image](inference_result_31M/train.png)|![image](inference_result_31M/test_1.png)|![image](inference_result_31M/test_2.png)|
|7M|![image](inference_result_7M/train.png)|![image](inference_result_7M/test_1.png)|![image](inference_result_7M/test_2.png)|
|1M|![image](inference_result_1M/train.png)|![image](inference_result_1M/test_1.png)|![image](inference_result_1M/test_2.png)|
|400k|![image](inference_result_400k/train.png)|![image](inference_result_400k/test_1.png)|![image](inference_result_400k/test_2.png)|

The results are promising, but the 400k model tent to be unstable in its prediction. The 1M seems to be a good trade-off between size and performance.

## 3 - Quantize the model (1M, INT8)
### First try: Post-training static quantization (PTQ)
First, I used the post-training static quantization method from PyTorch. The process involves the following steps:
1. Load the pre-trained model.
2. Prepare the model for quantization by inserting observers.
3. Calibrate the model using a representative dataset to collect statistics for quantization.
4. Convert the model to INT8 format.

The results where surprisingly excellent ! The quantized model is as good as the original model, as it can be seen on the following comparison between the FP32 1M model and the quantized INT8 1M model. 
![image](inference_result_1M/comparison_PTQ_int8.png)

### Second try: Quantization-aware training (QAT)
I also tried the quantization-aware training method.
The process involves the following steps:
1. Load the pre-trained model.
2. Prepare the model for quantization-aware training by inserting fake quantization modules.
3. Train the model from scratch and with the pre-trained weights
4. Convert the trained model to INT8 format.

The results for the "from scratch" training are not as good as the PTQ method, but still pretty good. The quantized model is slightly worse, specially on the test 1, where it recognize more clouds, but the results are still pretty good. In general, it tends to struggle to draw a clean outline of the clouds.
![image](inference_result_1M/comparison_QAT_FS_int8.png)

The results for the "with pre-trained weights" where better, but still not as good as the PTQ method. On the test 2, the model recognize more clouds. We can also see a little bit more noise in the prediction.
![image](inference_result_1M/comparison_QAT_FT_int8.png)

### Conclusion of the quantization part
The results are pretty surprising, as the PTQ method seems to be better, more stable and easier to implement than the QAT method. Here are my theories:
- The "from scratch" QAT can't be as good as the PTQ method, because during the training, the gradient is not as accurate due to the fake quantization modules.
- The "with pre-trained weights" is not as good as the PTQ method, because the fine-tuning process overfits the model and add some noise in the prediction. I tried with a sixth and a half of the epoch, but it didn't change much the results.

![image](inference_result_1M/comparison_all_int8.png)