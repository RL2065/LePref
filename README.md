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
   score_dict_multi = infer_all_conditions_multi_example(["<img1_obj_or_path>", "<img2_obj_or_path>", ...], "<prompt>", model, image_processor, tokenizer)
   ```
## Pairwise Accuracy of LeScore
| Method                      | Multi-dimension   | Human Annotation   |    Pick-a-Pic v1|   HPD v2 |   ImageRewardDB | Mean   |
|:----------------------------|:--------|:--------------|:----------------|---------:|----------------:|:-------|
| CLIPScore        |         | ✔             |            0.608|    0.712 |           0.543 | 0.628  |
| Aesthetic Score  |         | ✔             |            0.568|    0.726 |           0.574 | 0.623  |
| HPS v1              |         | ✔             | 0.667|    0.731 |           0.612 | 0.67   |
| HPS v2           |         | ✔             | 0.674|    0.833 |           0.657 | 0.721  |
| PickScore       |         | ✔             | 0.705|    0.798 |           0.629 | 0.711  |
| ImageReward    |         | ✔             | 0.611|    0.706 |           0.651 | 0.656  |
| MPS                   | ✔       | ✔             | / |    0.835 |           0.675 | /      |
| MPS_overall          |         | ✔             | 0.637|    0.838 |           0.663 | 0.713  |
| VP-Score     |         |               | 0.671|    0.794 |           0.663 | 0.709  |
| LeScore                     | ✔       |               | 0.674|    0.829 |           0.642 | 0.715  |

## Installation for LePref Annotation

First, create a virtual environment and install PyTorch:

```bash
conda create -n lepref python=3.10.16
conda activate lepref
pip install torch==2.5.1+cu124 torchaudio==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
```

Then, install vllm:
```bash
pip install vllm
```

## LePref Annotation 

For LePref Annotation , we adopt [Pixtral 12B](https://drive.google.com/file/d/1g3JnycSLlmKqIXGDjYYR5eoDxrowiGte/view?usp=drive_link), but you can choose any [vllm](https://docs.vllm.ai/en/latest/) supported MLLM models.

Use `MLLM_model_sampling_loader` to load your model and configure sampling parameters. This function initializes the model with specific memory limits and token configurations.

 ```python
 from annoutils import MLLM_model_sampling_loader, generate_request
 chat_func, sampling_params = MLLM_model_sampling_loader(
     n=1,
     temperature=0,
     model_name="mistralai/Pixtral-12B-2409"
 )
 ```

The `generate_request` function builds a list of messages based on the provided template type. Here is the template type and supported inputs and usage:

 | Type  | Isage    | Supported Input Forms |
 |:-:|:-:|:-:|
 | prompt_processing   | Processes a text-to-image prompt by splitting it into four distinct parts: image content, magic words, artistic style, and visual properties.           | Text prompt    |
 | prompt_NSFW_filtering   | Analyzes a text prompt to detect any NSFW (Not Safe For Work) content.       | Text prompt        |
 | prompt_classification   | Classifies a text-to-image prompt into one of the following categories: Abstract & Artistic, Animals & Plants, Characters, Objects & Food, or Scenes          | Text prompt |
 | captioning_coco_image	   | Generates a precise caption for an real image, focusing solely on the content.                  |Single image |
 | captioning_aes_image   | Provides a concise description of an image’s content along with a brief evaluation of its artistic style and notable visual properties          |Single image  |
 | annotation_alignment   | Evaluates how well each provided image aligns with a given text prompt.              | Text prompt and a list of images|
 | annotation_aesthetic   | Evaluates the aesthetic quality of AI-generated images with a given text prompt.           | Text prompt and a list of images|
 | annotation_fidelity  | Evaluates the aesthetic quality of AI-generated images with a given text prompt.           | Text prompt and a list of images|

```python
 messages = generate_request(
     template_type="annotation_alignment",
     prompt="<prompt>",
     image=["<img1_obj_or_path>", "<img2_obj_or_path>", ...]
 )
 ```

Call the chat function with the generated messages and sampling parameters to obtain a response:

 ```python
 response = chat_func(messages, sampling_params)
 print(response)
 ```
    
## Acknowledgments

We thank the authors of [ImageReward](https://github.com/kekewind/ImageReward), [HPS](https://github.com/tgxs002/align_sd), [HPS v2](https://github.com/tgxs002/HPSv2), [PickScore](https://github.com/yuvalkirstain/PickScore), and [MPS](https://github.com/Kwai-Kolors/MPS) for their code and papers, which have greatly contributed to our work.

