from torch import Tensor, nn
import fairseq
import torch
import torch.nn.functional as F
import speechbrain as sb

#import matplotlib.pyplot as plt

class HuBERTWrapper_extractor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ckpt_path = "models/facebook/HuBERT/hubert_base_ls960.pt"
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = models[0].feature_extractor
        self.model.requires_grad_(False)
        
    def forward(self, data: Tensor):
        #print(self.model)
        #print(data.shape)
        return self.model(data)

class HuBERTWrapper_extractor_all(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ckpt_path = "models/facebook/HuBERT/hubert_base_ls960.pt"
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = models[0].feature_extractor
        self.model._requires_grad=False

    def forward(self, data: Tensor):
        conv_layers = self.model
        #print(conv_layers)
        layers_list = [data.unsqueeze(1)]
        for f in conv_layers.conv_layers:
            #print(layers_list[-1].shape)
            #print(f)
            x_f = f(layers_list[-1])
            layers_list.append(x_f)
            #print(x_f.shape)
        #pad each tensor in layers_list to the size of the longest:
        pad_to = layers_list[1].shape[-1]
        out_layers_list = []
        for l in layers_list[1:]:
            padded_l = torch.nn.functional.pad(l,(0,pad_to-l.shape[-1]))
            #print("padded_l",padded_l.shape)
            out_layers_list.append(padded_l.permute(0,2,1))

        return out_layers_list

        

class HuBERTWrapper_full(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ckpt_path = "models/facebook/HuBERT/hubert_base_ls960.pt"

        models = fairseq.checkpoint_utils.load_model_ensemble([ckpt_path])
        full_model = models[0][0]
        full_model.features_only =True
        self.model = full_model

    def forward(self, data: Tensor):
        
        """
        my_output = None
        def my_hook(module_,input_,output_):
            nonlocal my_output
            my_output = output_

        a_hook = self.model.encoder.layers[6].final_layer_norm.register_forward_hook(my_hook)
        self.model(data)
        a_hook.remove()
        """
        my_output =self.model(data)
        return my_output['x']

class HuBERTWrapper_full_all(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ckpt_path = "models/facebook/HuBERT/hubert_base_ls960.pt"

        models = fairseq.checkpoint_utils.load_model_ensemble([ckpt_path])
        full_model = models[0][0]
        full_model.features_only =True
        self.model = full_model
        self.model._requires_grad=False

    def forward(self, data: Tensor):
        X = self.model.post_extract_proj(self.model.feature_extractor(data).permute(0,2,1))
        #print(X.shape)
        X = self.model.encoder.pos_conv(X.permute(0,2,1)).permute(0,2,1)
        #print(X.shape)
        layers_list = [X]
        for f in self.model.encoder.layers:
            #print(f)
            x_f = f(layers_list[-1])
            #print(x_f[0].shape)
            #print(x_f[1][1].shape)
            layers_list.append(x_f[0])
        
        #print(len(layers_list))
        #for l in layers_list:
        #    print(l.shape)
        return layers_list
