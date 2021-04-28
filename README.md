# Deep-Polarimetric-HDR-Reconstruction (DPHR)
Deep HDR Reconstruction using Polarimetric Cues from the Polarization Camera

<img src="https://user-images.githubusercontent.com/38761535/116477584-0fb4aa80-a83a-11eb-9a01-682f84ae97af.PNG" width="834" height="462">




## Requirements
* Python 3.8.5
* Pytorch 1.2.0
* torchvision 0.4.0
* OpenCV 4.1.2.30
* Numpy 1.19.1
* Pillow 6.1.0
* OpenEXR 1.3.2
* CUDA 10.2
* cuDNN 7.6.5
* NVIDIA GTX 1080 TI

```
pip install -r requirements.txt
```

## Usage
### Dataset
The collected EdPolCommunity dataset can be found in the dataset folder on Google Drive: [pol_outdoor1](https://drive.google.com/file/d/18nhczTSCFMB4_oUZZzyF_kHhqNCt8MGs/view?usp=sharing), [pol_outdoor2](https://drive.google.com/file/d/1za16n_CeqPrNUAkFdxjT2Hf_bTB3cthi/view?usp=sharing)

### Pretrained model
The pretrained model checkpoints can be found in the checkpoints folder on [Google Drive](https://drive.google.com/file/d/13RZGW3BnX4O5liuYHMHBdYt9srewLMlS/view?usp=sharing)

### Inference
Sample code for inference using the DPHR model
```
python sample_code.py --i0 PATH/TO/POL0folder --i45 PATH/TO/POL45folder --i90 PATH/TO/POL90folder --i135 PATH/TO/POL135folder --out_dir PATH/TO/OUTfolder/im_{idx}.png --weights model/DPHR_checkpoint.pt 
```
Training code will be added soon.

### Supplementary materials
We provide more visual comparisons in the supplementary material PDF. Namely, we provide more qualitative results on: 1) Evaluation of the HDR formulation. 2) Comparison with state-of-the-art methods.

## Reference
If you find this work useful in your research, please cite:
