# Show, Match and Segment: Joint Weakly Supervised Learning of Semantic Matching and Object Co-segmentation

This repository contains the source code for the paper Show, Match and Segment: Joint Weakly Supervised Learning of Semantic Matching and Object Co-segmentation.

<img src="img/teaser.png" width="1000">

## Abstract
We present an approach for jointly matching and segmenting object instances of the same category within a collection of images. In contrast to existing algorithms that tackle the tasks of semantic matching and object co-segmentation in isolation, our method exploits the complementary nature of the two tasks. The key insights of our method are two-fold. First, the estimated dense correspondence fields from semantic matching provide supervision for object co-segmentation by enforcing consistency between the predicted masks from a pair of images. Second, the predicted object masks from object co-segmentation in turn allow us to reduce the adverse effects due to background clutters for improving semantic matching. Our model is end-to-end trainable and does not require supervision from manually annotated correspondences and object masks. We validate the efficacy of our approach on five benchmark datasets: TSS, Internet, PF-PASCAL, PF-WILLOW, and SPair-71k, and show that our algorithm performs favorably against the state-of-the-art methods on both semantic matching and object co-segmentation tasks.

## Citation
If you find our code useful, please consider citing our work using the following bibtex:
```
@article{MaCoSNet,
    title={Show, Match and Segment: Joint Weakly Supervised Learning of Semantic Matching and Object Co-segmentation},
    author={Chen, Yun-Chun and Lin, Yen-Yu and Yang, Ming-Hsuan and Huang, Jia-Bin},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI)},
    year={2020}
}

@inproceedings{WeakMatchNet,
  title={Deep Semantic Matching with Foreground Detection and Cycle-Consistency},
  author={Chen, Yun-Chun and Huang, Po-Hsiang and Yu, Li-Yu and Huang, Jia-Bin and Yang, Ming-Hsuan and Lin, Yen-Yu},
  booktitle={Asian Conference on Computer Vision (ACCV)},
  year={2018}
}
```

## Environment
 - Install Anaconda Python3.7
 - This code is tested on NVIDIA V100 GPU with 16GB memory
 
``` 
pip install -r requirements.txt
```

## Dataset
 - Please download the [PF-PASCAL](http://www.di.ens.fr/willow/research/proposalflow/dataset/PF-dataset-PASCAL.zip), [PF-WILLOW](http://www.di.ens.fr/willow/research/proposalflow/dataset/PF-dataset.zip), [SPair-71k] (http://cvlab.postech.ac.kr/research/SPair-71k/), [TSS](https://drive.google.com/file/d/0B-VxeI7PlJE1U3FyTGVpbUFtcjg/view?usp=sharing), and [Internet](http://people.csail.mit.edu/mrub/ObjectDiscovery/ObjectDiscovery-data.zip) datasets
 - Please modify the variable `DATASET_DIR` in `config.py` 
 - Please modify the variable `CSV_DIR` in `config.py`



## Training
 - You may determine which dataset to be the `training set` by changing the $DATASET variable in train.sh
 - You may change the $BATCH_SIZE variable in `train.sh` to a suitable value based on the GPU memory
 - The trained model will be saved under the `trained_models` folder
 
``` 
sh train.sh
```


## Evaluation
 - You may determine which dataset to be evaluated by changing the $DATASET variable in eval.sh
 - You may change the $BATCH_SIZE variable in `eval.sh` to a suitable value based on the GPU memory
 
``` 
sh eval.sh
```

## Acknowledgement
 - This code is heavily borrowed from [Rocco et al.](https://github.com/ignacio-rocco/weakalign)
