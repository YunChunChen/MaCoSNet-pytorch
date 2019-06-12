# Show, Match and Segment: Joint Learning of Semantic Matching and Object Co-segmentation

This repository contains the source code for the paper Show, Match and Segment: Joint Learning of Semantic Matching and Object Co-segmentation.

## Citation
If you find our code useful, please consider citing our work using the bibtex:
```
@article{MaCoSNet,
    title={Show, Match and Segment: Joint Learning of Semantic Matching and Object Co-segmentation},
    author={Chen, Yun-Chun and Lin, Yen-Yu and Yang, Ming-Hsuan and Huang, Jia-Bin},
    journal={arXiv},
    year={2019}
}
```

## Enviroment
 - Install Anaconda Python3.7
 
``` 
pip install -r requirements.txt
```

## Training
 - This code is tested on NVIDIA V100 GPU with 16GB memory
 - You may change the number of batch size based on the GPU memory in the train.sh $BATCH_SIZE variable
 - The trained model will be saved under the *trained_models* folder
 
``` 
sh train.sh
```


## Evaluation
 
``` 
sh eval.sh
```
