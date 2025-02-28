# LePref and LeScore

This repository contains the code and models for the paper *Leveraging Multi-Modal Large Language Models for Human Preference Prediction in Text-to-Image Generation*.

## Installation for LeScore Inference

First, create a virtual environment and install PyTorch:

```bash
conda create -n lescore python=3.9.21
conda activate lescore
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Then, install the required packages:

```bash
pip install -r requirements.txt
```

## Inference with LeScore

For inference with LeScore, follow these steps:

1. Download the Checkpoint
   First, download the LeScore checkpoint from [this link](https://drive.google.com/file/d/1g3JnycSLlmKqIXGDjYYR5eoDxrowiGte/view?usp=drive_link), then move the checkpoint file to the `outputs` directory.

2. Inference for a Single Image 
   To infer the score for a single image, use the following example code:
   
   ```python
   from utils import load_pretrained_model, infer_all_conditions_one_sample, infer_all_conditions_multi_example

   model, image_processor, tokenizer = load_pretrained_model()
   score_dict = infer_all_conditions_one_sample("<img1_obj_or_path>", "<prompt>", model, image_processor, tokenizer)
   ```
   
3. Inference for Multiple Examples  
   If you want to infer scores for multiple examples using the same prompt:
   
   ```python
   score_dict_multi = infer_all_conditions_multi_example("<img1_obj_or_path>", "<prompt>", model, image_processor, tokenizer)
   ```

## Acknowledgments

We thank the authors of [ImageReward](https://github.com/kekewind/ImageReward), [HPS](https://github.com/tgxs002/align_sd), [HPS v2](https://github.com/tgxs002/HPSv2), [PickScore](https://github.com/yuvalkirstain/PickScore), and [MPS](https://github.com/Kwai-Kolors/MPS) for their code and papers, which have greatly contributed to our work.
