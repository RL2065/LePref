# 🎨 LePref and LeScore

> This repository contains the code and models for the paper *Leveraging Multi-Modal Large Language Models for Human Preference Prediction in Text-to-Image Generation*.

## 📋 Table of Contents
- [Installation](#installation)
  - [LeScore Inference](#installation-for-lescore-inference)
  - [LePref Annotation](#installation-for-lepref-annotation)
- [Usage](#usage)
  - [LeScore Inference](#inference-with-lescore)
  - [LePref Annotation](#lepref-annotation)
- [Performance](#performance)
- [Acknowledgments](#acknowledgments)

## 🛠️ Installation

### Installation for LeScore Inference

1. Create a virtual environment and install PyTorch:
```bash
conda create -n lescore python=3.9.21
conda activate lescore
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Installation for LePref Annotation

1. Create a virtual environment and install PyTorch:
```bash
conda create -n lepref python=3.10.16
conda activate lepref
pip install torch==2.5.1+cu124 torchaudio==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
```

2. Install vllm:
```bash
pip install vllm
```

## 🚀 Usage

### Inference with LeScore

1. **Download the Checkpoint**
   - Download the LeScore checkpoint from [this link](https://drive.google.com/file/d/1g3JnycSLlmKqIXGDjYYR5eoDxrowiGte/view?usp=drive_link)
   - Move the checkpoint file to the `outputs` directory

2. **Single Image Inference**
```python
from utils import load_pretrained_model, infer_all_conditions_one_sample, infer_all_conditions_multi_example

model, image_processor, tokenizer = load_pretrained_model()
score_dict = infer_all_conditions_one_sample("<img1_obj_or_path>", "<prompt>", model, image_processor, tokenizer)
```

3. **Multiple Examples Inference**
```python
score_dict_multi = infer_all_conditions_multi_example(
    ["<img1_obj_or_path>", "<img2_obj_or_path>", ...], 
    "<prompt>", 
    model, 
    image_processor, 
    tokenizer
)
```

### LePref Annotation

We use [Pixtral 12B](https://drive.google.com/file/d/1g3JnycSLlmKqIXGDjYYR5eoDxrowiGte/view?usp=drive_link), but you can choose any [vllm](https://docs.vllm.ai/en/latest/) supported MLLM models.

1. **Load Model and Configure Sampling**
```python
from annoutils import MLLM_model_sampling_loader, generate_request

chat_func, sampling_params = MLLM_model_sampling_loader(
    n=1,
    temperature=0,
    model_name="mistralai/Pixtral-12B-2409"
)
```

2. **Generate Requests**
```python
messages = generate_request(
    template_type="annotation_alignment",
    prompt="<prompt>",
    image=["<img1_obj_or_path>", "<img2_obj_or_path>", ...]
)
```

3. **Get Response**
```python
response = chat_func(messages, sampling_params)
print(response)
```

## 📊 Supported Template Types

| Type | Description | Input Format |
|------|-------------|--------------|
| `prompt_processing` | Splits text-to-image prompt into content, magic words, style, and properties | Text prompt |
| `prompt_NSFW_filtering` | Detects NSFW content in text prompts | Text prompt |
| `prompt_classification` | Classifies prompts into categories (Abstract, Animals, Characters, etc.) | Text prompt |
| `captioning_coco_image` | Generates precise captions for COCO images | Single image |
| `captioning_aes_image` | Describes content and evaluates style for LAION Aes 6.5+ images | Single image |
| `annotation_alignment` | Evaluates alignment quality of AI-generated images | Text + Image list |
| `annotation_aesthetic` | Evaluates aesthetic quality of AI-generated images | Text + Image list |
| `annotation_fidelity` | Evaluates fidelity quality of AI-generated images | Text + Image list |

## 📈 Performance

### Pairwise Accuracy of LeScore

| Method | Multi-dimension | Human Annotation | Pick-a-Pic v1 | HPD v2 | ImageRewardDB | Mean |
|:-------|:---------------|:-----------------|:-------------|:------|:--------------|:-----|
| CLIPScore | | ✔ | 0.608 | 0.712 | 0.543 | 0.628 |
| Aesthetic Score | | ✔ | 0.568 | 0.726 | 0.574 | 0.623 |
| HPS v1 | | ✔ | 0.667 | 0.731 | 0.612 | 0.67 |
| HPS v2 | | ✔ | 0.674 | 0.833 | 0.657 | 0.721 |
| PickScore | | ✔ | 0.705 | 0.798 | 0.629 | 0.711 |
| ImageReward | | ✔ | 0.611 | 0.706 | 0.651 | 0.656 |
| MPS | ✔ | ✔ | / | 0.835 | 0.675 | / |
| MPS_overall | | ✔ | 0.637 | 0.838 | 0.663 | 0.713 |
| VP-Score | | | 0.671 | 0.794 | 0.663 | 0.709 |
| LeScore | ✔ | | 0.674 | 0.829 | 0.642 | 0.715 |

## 🙏 Acknowledgments

We thank the authors of:
- [ImageReward](https://github.com/THUDM/ImageReward)
- [HPS](https://github.com/tgxs002/align_sd)
- [HPS v2](https://github.com/tgxs002/HPSv2)
- [PickScore](https://github.com/yuvalkirstain/PickScore)
- [MPS](https://github.com/Kwai-Kolors/MPS)

Their code and papers have greatly contributed to our work.

