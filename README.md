# R3DG
The code for R3DG: Retrieve, Rank and Reconstruction with Different Granularities for Multimodal Sentiment Analysis. This paper has been accepted by [Research](https://spj.science.org/doi/10.34133/research.0729). 
### The Framework of R3DG:
![image](https://github.com/YetZzzzzz/R3DG/blob/main/framework.png)
Figure 1:  Illustration of our R3DG framework. It consists of 3 submodules. The retrieve and rank module acquires representations from each modality, employing adaptive pooling to capture various granularities of video and audio modalities. It ranks and selects the top-k representations most similar to textual features for fusion. The reconstruction module reconstructs the original audio and video representations through the fused local audio and video representations. Lastly, the multimodal fusion module integrates the extracted features from each modality for prediction. Different colors of cubes in the video and audio modalities denote varying granularities of local representations.


### Datasets:
**Please move the following datasets into directory ./datasets/.**

The aligned CMU-MOSI and CMU-MOSEI datasets can be downloaded according to [MAG](https://github.com/WasifurRahman/BERT_multimodal_transformer) and [BERT_MulT](https://github.com/WasifurRahman/BERT_multimodal_transformer/blob/master/datasets/download_datasets.sh). 

The unaligned CMU-MOSI and CMU-MOSEI datasets can be downloaded according to [MulT](https://github.com/yaohungt/Multimodal-Transformer/tree/master).

Please put the aligned and unaligned datasets to ./dataset/aligned/ and ./dataset/unaligned/ seperately.

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
Downlaod the [BERT-base](https://huggingface.co/google-bert/bert-base-uncased/tree/main) , and put into directory ./pretrained_models/BERT_EN/.

### Run R3DG
For MOSI: 
```
python3 main_R3DG.py --text_seq_length=50 --audio_seq_length=500 --visual_seq_length=375 --TEXT_DIM=768 --ACOUSTIC_DIM=5 --VISUAL_DIM=20 --alignment=unalign --dataset=mosi --local_as=10 --train_batch_size=64
python3 main_R3DG.py --text_seq_length=50 --audio_seq_length=50 --visual_seq_length=50 --TEXT_DIM=768 --ACOUSTIC_DIM=74 --VISUAL_DIM=47 --alignment=align --dataset=mosi --local_vs=5 --learning_rate=5e-5
```
For MOSEI: 
```
python3 main_R3DG.py --text_seq_length=50 --audio_seq_length=50 --visual_seq_length=50 --TEXT_DIM=768 --ACOUSTIC_DIM=74 --VISUAL_DIM=35 --alignment=align --dataset=mosei --learning_rate=2e-5 --layers=4
python3 main_R3DG.py --text_seq_length=50 --audio_seq_length=500 --visual_seq_length=375 --TEXT_DIM=768 --ACOUSTIC_DIM=74 --VISUAL_DIM=35 --alignment=unalign --dataset=mosei --alpha=0.3 --train_batch_size=64
```

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

### Acknowledgement
Thanks to  [MIB](https://github.com/TmacMai/Multimodal-Information-Bottleneck) , [MAG](https://github.com/WasifurRahman/BERT_multimodal_transformer),  [MCL](https://github.com/TmacMai/Multimodal-Correlation-Learning), [HKT](https://github.com/matalvepu/HKT), [LFMIM](https://github.com/sunjunaimer/LFMIM) and [LFMIM](https://github.com/sunjunaimer/LFMIM) for their great help to our codes and research. 



