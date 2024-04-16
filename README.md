<div align="center">

# <b>DiffSHEG</b>: A Diffusion-Based Approach for Real-Time Speech-driven Holistic 3D Expression and Gesture Generation (CVPR 2024 Official Repo)

[Junming Chen](https://jeremycjm.github.io)<sup>&dagger;,1,2</sup>, [Yunfei Liu](http://liuyunfei.net/)<sup>2</sup>, [Jianan Wang](https://scholar.google.com/citations?user=mt5mvZ8AAAAJ&hl=en&inst=1381320739207392350)<sup>2</sup>, [Ailing Zeng](https://ailingzeng.site/)<sup>2</sup>, [Yu Li](https://yu-li.github.io/)<sup>*,2</sup>, [Qifeng Chen](https://cqf.io)<sup>1</sup>

<p><sup>1</sup>HKUST &nbsp;&nbsp;<sup>2</sup>International Digital Economy Academy (IDEA) &nbsp;&nbsp;
<br><sup>*</sup>Corresponding authors &nbsp;&nbsp;<sup>&dagger;</sup>Work done during an internship at IDEA<p>

### [Project Page](https://jeremycjm.github.io/proj/DiffSHEG/) · [Paper](https://arxiv.org/abs/2401.04747) · [Video](https://www.youtube.com/watch?v=HFaSd5do-zI)

</div>

# Environment
We have tested on Ubuntu 18.04 and 20.04.
## Option 1: conda install
```
conda env create -f environment.yml
conda activate diffsheg
```
## Option 2: pip install
```
conda create -n "diffsheg" python=3
conda activate diffsheg
pip install torch torchvision torchaudio
pip install -U openmim
mim install "mmcv<2.0.0"
pip install -r requirements.txt
```
## Untar data.tar.gz for data statistics
```
tar zxvf data.tar.gz
```

# Checkpoints
[Google Drive](https://drive.google.com/file/d/1JPoMOcGDrvkFt7QbN6sEyYAPOOWkVN0h/view)

# Inference on a Custom Audio
First specify the '--test_audio_path' argument to your test audio path in the following mentioned bash files. Note that the audio should be a .wav file.
Use model trained on BEAT dataset:
```bash inference_custom_audio_beat.sh```
Use model trained on SHOW dataset:
```bash inference_custom_audio_talkshow.sh```

# Visualization
After running under the test or test-custom-audio mode, the Gesture and Expression results will be saved in the ./results directory.
## BEAT
1. Open the data2video.blend with latest Blender on your local computer.
2. Specify the audio, BVH (for gesture), JSON (for expression), and video saving path in the transcript in Blender.
3. (Optional) Click Window --> Toggle System Console to check the visulization progress.
4. Run the script in Blender.
## SHOW
1. ```cd  A_TalkSHOW_ori```
2. Specify the gesture results directory path ```--gesture_path``` and expression results directory path ```--face_path``` in ```visualise_ddpm.sh``` 
3. ```bash visualise_ddpm.sh``` 


# Citation
If you use our code or find this repo useful, please consider cite our paper:
```
@inproceedings{ChenDiffsheg2024,
  title     = {DiffSHEG: A Diffusion-Based Approach for Real-Time Speech-driven Holistic 3D Expression and Gesture Generation},
  author    = {Chen, Junming and Liu, Yunfei and Wang, Jianan and Zeng, Ailing and Li, Yu and Chen, Qifeng},
  booktitle = {CVPR},
  year      = {2024}
}
```