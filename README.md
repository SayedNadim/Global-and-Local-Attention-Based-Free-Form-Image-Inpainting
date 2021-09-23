[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/global-and-local-attention-based-free-form/image-inpainting-on-places2)](https://paperswithcode.com/sota/image-inpainting-on-places2?p=global-and-local-attention-based-free-form)

### <s>I am currently refactoring the codes for the latest version of PyTorch. I will update the codes and upload the pretrained models (e.g. Places and CelebA) soon. Apologies for the inconvenience.</s>
### Please checkout "sensor_version" for Places2 weights. Please let me know if you face any issue.

This is the official implementation of the paper "Global and Local Attention-Based Free-Form Image Inpainting" published in Sensors ([paper](https://www.mdpi.com/1424-8220/20/11/3204)). Currently we are reformatting the codes. We will upload the pretrained models soon.

### Prerequisite
- Python3
- PyTorch 1.0+ (The code works up to PyTorch 1.4. There seems to be an auto-grad problem with PyTorch 1.5. I will update the code for PyTorch 1.5 after finding the underlying issue.)
- Torchvision 0.2+
- PyYaml


### Citation
If you find our paper and code beneficial for your work, please consider citing us!
<br>
```
@article{uddin2020global,
  title={Global and Local Attention-Based Free-Form Image Inpainting},
  author={Uddin, SM and Jung, Yong Ju},
  journal={Sensors},
  volume={20},
  number={11},
  pages={3204},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
</br>

### How to train
- Set directory path in "configs/config.yaml". 
-- Set dataset name, if needed. 
-- If the dataset has subfolders, set "data_with_subfolder" to "True".
- Run
``` python train.py --config configs/config.yaml ```
- To resume, set "resume" to True in "configs/config.yaml". Currently it overwrites the previous checkpoints. Updated code will have checkpoints listed.
- To view training, run <br>
```tensorboard --logdir checkpoints/DATASET_NAME/hole_benchmark ```

### How to test
- Modify "test_single.py" as per need and run.
- Bulk testing code will be uploaded soon.
- Pretrained models will be uploaded soon. 

### Some Results
- Places dataset
![alt text](img/places.png)
- ImageNet dataset
![alt text](img/imagenet.png)
- CelebA dataset
![alt text](img/celeba.png)
- Ablation study of the modules
![alt text](img/ablation.png)

#### Acknowledgement
- Code base: This code is heavily relied on [this repo](https://github.com/daa233/generative-inpainting-pytorch). Kudus!!!
