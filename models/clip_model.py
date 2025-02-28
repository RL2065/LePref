from dataclasses import dataclass

from transformers import CLIPModel as HFCLIPModel
from transformers import CLIPConfig
from transformers import AutoTokenizer


from .cross_modeling import Cross_model
from .base_model import BaseModelConfig


from typing import Any, Optional, Tuple, Union


import torch
from torch import nn, einsum
import torch.nn.functional as F

import gc



@dataclass
class ClipModelConfig(BaseModelConfig):
    _target_: str = "trainer.models.clip_model.CLIPModel"
    pretrained_model_name_or_path: str ="laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

class XCLIPModel(HFCLIPModel):
    def __init__(self, config: CLIPConfig):
        super().__init__(config)
    
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:

        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # pooled_output = text_outputs[1]
        # text_features = self.text_projection(pooled_output)
        last_hidden_state = text_outputs[0]
        text_features = self.text_projection(last_hidden_state)

        pooled_output = text_outputs[1]
        text_features_EOS = self.text_projection(pooled_output)


        # del last_hidden_state, text_outputs
        # gc.collect()

        return text_features, text_features_EOS

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # pooled_output = vision_outputs[1]  # pooled_output
        # image_features = self.visual_projection(pooled_output)
        last_hidden_state = vision_outputs[0]
        image_features = self.visual_projection(last_hidden_state)

        return image_features






class CLIPModel(nn.Module):
    def __init__(self, cfg=ClipModelConfig, by_ckpt = None):
        super().__init__()
        self.model = XCLIPModel.from_pretrained(cfg.pretrained_model_name_or_path)

                
        self.cross_models = nn.ModuleList([
            Cross_model(dim=1024, layer_num=4, heads=16),  # 0: aesthetic
            Cross_model(dim=1024, layer_num=4, heads=16),  # 1: fidelity
            Cross_model(dim=1024, layer_num=4, heads=16),  # 2: alignment
        ])
        
        if by_ckpt:
            self.load_state_dict(torch.load(by_ckpt))
        
             
    def get_text_features(self, *args, **kwargs):
        return self.model.get_text_features(*args, **kwargs)

    def get_image_features(self, *args, **kwargs):
        return self.model.get_image_features(*args, **kwargs)

    def forward(self, text_inputs=None, image_inputs=None, condition_inputs=None):
        outputs = ()
        text_f, text_EOS = self.model.get_text_features(text_inputs)  # (B, n_p, 1024)
        text_EOS = text_EOS.repeat_interleave(3, dim=0)
        outputs += (text_EOS,)

        image_f = self.model.get_image_features(image_inputs.half())  # (2B, 257, 1024)
        condition_f, _ = self.model.get_text_features(condition_inputs) # (3B, n_c, 1024)
        
        # [3B, n_c, d] -> [B, 3, n_c, d] -> [3, B, n_c, d]
        B3 = condition_f.size(0)
        B_ = B3 // 3
        condition_f_reshaped = condition_f.view(B_, 3, *condition_f.shape[1:]) 
       
        condition_f_reshaped = condition_f_reshaped.permute(1, 0, *range(2, condition_f_reshaped.dim()))  
        #  condition_f_reshaped : [3, B, n_c, d]

        sim0_list = []
        sim1_list = []
        
        for i in range(3):
            condition_f_single = condition_f_reshaped[i]   # [B, n_c, d]
            condition_f_norm = F.normalize(condition_f_single, p=2, dim=-1)
            text_f_norm       = F.normalize(text_f,         p=2, dim=-1)

        
            sim_text_condition = torch.einsum('b c d, b p d -> b c p',
                                            condition_f_norm, text_f_norm)  # (B, n_c, n_p)
            sim_text_condition = torch.max(sim_text_condition, dim=1, keepdim=True)[0]  # (B,1,n_p)
            
            mask = torch.where(sim_text_condition > 0.3, 0, float('-inf'))  # (B,1,n_p)
            mask = mask.repeat(1, image_f.shape[1], 1)                      # (B,257,n_p)

            bc = image_f.shape[0] // 2  # = B
            sim0 = self.cross_models[i](image_f[:bc],   text_f, mask.half())  # -> (B, 1, d)
            sim1 = self.cross_models[i](image_f[bc:],   text_f, mask.half())  # -> (B, 1, d)

            sim0_list.append(sim0[:,0,:])  # [B, d]
            sim1_list.append(sim1[:,0,:])  # [B, d]

        sim0_stack = torch.stack(sim0_list, dim=0)  # [3, B, d]
        sim1_stack = torch.stack(sim1_list, dim=0)  # [3, B, d]

        # permute(1,0,2) => [B, 3, d] then  view => [3B, d]
        sim0_reordered = sim0_stack.permute(1,0,2).reshape(-1, sim0_stack.size(-1))  # [3B, d]
        sim1_reordered = sim1_stack.permute(1,0,2).reshape(-1, sim1_stack.size(-1))  # [3B, d]
        outputs += (sim0_reordered,sim1_reordered,)
        return outputs

    @property
    def logit_scale(self):
        return self.model.logit_scale

    def save(self, path):
        self.model.save_pretrained(path)

