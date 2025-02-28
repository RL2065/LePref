import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, AutoTokenizer
from datasets import load_from_disk
from models.clip_model import CLIPModel

def load_pretrained_model(by_ckpt='outputs/LeScore.pt'):
    model = CLIPModel(by_ckpt=by_ckpt)
    model.eval().to("cuda")
    
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    image_processor = CLIPImageProcessor.from_pretrained(processor_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(processor_name_or_path, trust_remote_code=True)
    
    return model,image_processor,tokenizer


@torch.no_grad()
def infer_all_conditions_one_sample(image, prompt, moe_model, clip_processor, tokenizer, device='cuda'):
    """
    对单张图片同时生成三个 condition 的结果，返回一个字典，
    字典键为 'aesthetic'、'fidelity'、'alignment'，值为对应的图像分数。
    """
    def _process_image(image):
        if isinstance(image, dict):
            image = image["bytes"]
        if isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        if isinstance(image, str):
            image = Image.open(image)
        return clip_processor(image.convert("RGB"), return_tensors="pt")["pixel_values"]

    def _tokenize(text):
        return tokenizer(
            text,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
    image_input = _process_image(image).to(device)
    text_input = _tokenize(prompt).to(device)

    condition_content_dict = {
        "aesthetic": "light, color, clarity, tone, style, ambiance, artistry.",
        "fidelity": "shape, face, hair, hands, limbs, structure, instance, texture.",
        "alignment": "quantity, attributes, position, number, location ;"
    }


    text_f, text_features = moe_model.model.get_text_features(text_input)
    image_f = moe_model.model.get_image_features(image_input)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    results = {}

    for idx, cond in enumerate(["aesthetic", "fidelity", "alignment"]):
        condition_text = condition_content_dict[cond]
        condition_batch = _tokenize(condition_text).repeat(text_input.shape[0], 1).to(device)
        condition_f, _ = moe_model.model.get_text_features(condition_batch)
        condition_f_norm = F.normalize(condition_f, p=2, dim=-1)
        text_f_norm = F.normalize(text_f, p=2, dim=-1)
        sim_text_condition = torch.einsum('b c d, b p d -> b c p', condition_f_norm, text_f_norm)
        sim_text_condition, _ = torch.max(sim_text_condition, dim=1, keepdim=True)
        mask = torch.where(sim_text_condition > 0.3, 0, float('-inf'))
        mask = mask.repeat(1, image_f.shape[1], 1)


        cross_model = moe_model.cross_models[idx]
        image_features = cross_model(image_f, text_f, mask)[:, 0, :]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        score = moe_model.logit_scale.exp() * text_features @ image_features.T
        results[cond] = score[0]
    results['overall'] = (results["aesthetic"]+results["fidelity"]+results["alignment"])/3
    return results

def infer_all_conditions_multi_example(images, prompt, moe_model, clip_processor, tokenizer):

    all_scores = {"aesthetic": [], "fidelity": [], "alignment": [], "overall": []}

    for image in images:
        score_dict = infer_all_conditions_one_sample(image, prompt, moe_model, clip_processor, tokenizer)
        for cond in all_scores:
            all_scores[cond].append(score_dict[cond])

    final_probs = {}
    for cond, scores in all_scores.items():
        scores_tensor = torch.stack(scores, dim=0)  
        probs = torch.softmax(scores_tensor, dim=0)
        final_probs[cond] = probs.cpu().tolist()

    return final_probs

