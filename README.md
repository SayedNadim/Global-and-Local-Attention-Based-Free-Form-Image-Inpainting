[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/global-and-local-attention-based-free-form/image-inpainting-on-places2)](https://paperswithcode.com/sota/image-inpainting-on-places2?p=global-and-local-attention-based-free-form)

This is the official implementation of the paper "Global and Local Attention-Based Free-Form Image Inpainting" published in Sensors ([paper](https://www.mdpi.com/1424-8220/20/11/3204)). Currently we are reformatting the codes. We will upload the pretrained models soon.

### Prerequisite
- Python3
- PyTorch 1.0+
- Torchvision 0.2+
- PyYaml

### How to train
- Set directory path in "configs/config.yaml". 
-- Set dataset name, if needed. 
-- If the dataset has subfolders, set "data_with_subfolder" to "True".
- Checkpoints can be found under "scripts" folder.
-- To resume, set "resume" to True in "configs/config.yaml". Currently it overwrites the previous checkpoints. Updated code will have checkpoints listed.
- To view training, run - "tensorboard --logdir scripts/checkpoints/DATASET_NAME/hole_benchmark".

### How to test
- Modify "test_single.py" as per need and run.
- I will upload bulk test code soon.
- Pretrained models will be uploaded soon. 

### Some Results






### Citation
If you find our paper and code beneficial for your work, please consider citing us!

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

#### Acknowledgement
- Code base: This code is heavily relied on [this repo](https://github.com/daa233/generative-inpainting-pytorch). I can not thank enough.
