# R3DG
The code for R3DG: Retrieve, Rank and Reconstruction with Different Granularities for Multimodal Sentiment Analysis. This paper has been accepted by [Research](https://spj.science.org/doi/10.34133/research.0729). 
### The Framework of R3DG:
![image](https://github.com/YetZzzzzz/R3DG/blob/main/framework.png)
Figure 1:  Illustration of our R3DG framework. It consists of 3 submodules. The retrieve and rank module acquires representations from each modality, employing adaptive pooling to capture various granularities of video and audio modalities. It ranks and selects the top-k representations most similar to textual features for fusion. The reconstruction module reconstructs the original audio and video representations through the fused local audio and video representations. Lastly, the multimodal fusion module integrates the extracted features from each modality for prediction. Different colors of cubes in the video and audio modalities denote varying granularities of local representations.


### Datasets:
**Please move the following datasets into directory ./datasets/.**

The CMU-MOSI and CMU-MOSEI datasets can be downloaded according to [MAG](https://github.com/WasifurRahman/BERT_multimodal_transformer) and [MulT](https://github.com/yaohungt/Multimodal-Transformer/tree/master). 

For UR-FUNNY and MUStARD, the dataset can be downloaded according to [HKT](https://github.com/matalvepu/HKT/blob/main/dataset/download.txt) through:
```
Download Link of UR-FUNNY:  https://www.dropbox.com/s/5y8q52vj3jklwmm/ur_funny.pkl?dl=1
Download Link of MUsTARD: https://www.dropbox.com/s/w566pkeo63odcj5/mustard.pkl?dl=1
```
Please rename the files as ur_funny.pkl and mustard.pkl, and move them into the directory ./datasets/.

For CHERMA dataset, you can download from [LFMIM](https://github.com/sunjunaimer/LFMIM) through: 
```
https://pan.baidu.com/s/10PoJcXMDhRg4fzsq96A7rQ
Extraction code: CHER
```
Please put the files into directory ./datasets/CHERMA0723/.

### Prerequisites:
```
* Python 3.8.10
* CUDA 11.5
* pytorch 1.12.1+cu113
* sentence-transformers 3.1.1
* transformers 4.30.2
```
**Note that the torch version can be changed to your cuda version, but please keep the transformers==4.30.2 as some functions will change in later versions**

### Pretrained model:
Downlaod the [BERT-base](https://huggingface.co/google-bert/bert-base-uncased/tree/main) , and put into directory ./BERT-EN/.

### Citation:
Please cite our paper if you find our work useful for your research:
```
@article{yan2025r3dg,
title = {R3DG: Retrieve, Rank, and Reconstruction with Different Granularities for Multimodal Sentiment Analysis},
author = {Yan Zhuang  and Yanru Zhang  and Jiawen Deng  and Fuji Ren },
journal = {Research},
volume = {8},
number = {},
pages = {0729},
year = {2025},
doi = {10.34133/research.0729},
}
```


