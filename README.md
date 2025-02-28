# LePref and LeScore
This repository contains the code and model for the paper [Leveraging Multi-Modal Large Language Models for Human Preference Prediction in Text-to-Image Generation]
## Installation for LeScore Inference
Create a virual env and download torch:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install the requirements:
```bash
pip install -r requirements.txt
```

## Inference with LeScore
We display here an example for running inference with LeScore:
```python

```

## Download the LeScore checkpoint
https://drive.google.com/file/d/1g3JnycSLlmKqIXGDjYYR5eoDxrowiGte/view?usp=drive_link
(Move the checkpoint file to `outputs`)


## Acknowledgments
We thank the authors of [ImageReward](https://github.com/kekewind/ImageReward), [HPS](https://github.com/tgxs002/align_sd), [HPS v2](https://github.com/tgxs002/HPSv2), [PickScore](https://github.com/yuvalkirstain/PickScore)  and [MPS](https://github.com/Kwai-Kolors/MPS) for their codes and papers, which greatly contributed to our work.
