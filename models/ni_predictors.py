import torch
import torch.nn.functional as F
from torch import Tensor, nn
try: #look in two places for the HuBERT wrapper
    from models.huBERT_wrapper import HuBERTWrapper_full,HuBERTWrapper_extractor,HuBERTWrapper_full_all
    #from models.wav2vec2_wrapper import Wav2Vec2Wrapper_no_helper,Wav2Vec2Wrapper_encoder_only
except:
    from huBERT_wrapper import HuBERTWrapper_full,HuBERTWrapper_extractor,HuBERTWrapper_full_all
    #from wav2vec2_wrapper import Wav2Vec2Wrapper_no_helper,Wav2Vec2Wrapper_encoder_only
from speechbrain.processing.features import spectral_magnitude,STFT

class PoolAttFF(torch.nn.Module):
    '''
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    '''         
    def __init__(self, dim_head_in):
        super().__init__()
        
        self.linear1 = nn.Linear(dim_head_in, 2*dim_head_in)
        self.linear2 = nn.Linear(2*dim_head_in, 1)
        
        self.linear3 = nn.Linear(dim_head_in, 1)
        
        self.activation = F.relu
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: Tensor):

        att = self.linear2(self.dropout(self.activation(self.linear1(x))))
        att = att.transpose(2,1)
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, x) 
        x = x.squeeze(1)
        
        x = self.linear3(x)
        
        return x  
        

class HuBERTMetricPredictorEncoderMulti(nn.Module):
    """Metric estimator for enhancement training.

    Consists of:
     * four 2d conv layers
     * channel averaging
     * three linear layers

    Arguments
    ---------
    kernel_size : tuple
        The dimensions of the 2-d kernel used for convolution.
    base_channels : int
        Number of channels used in each conv layer.
    """

    def __init__(
        self, dim_extractor=512, hidden_size=512//2, activation=nn.LeakyReLU,
    ):
        super().__init__()

        self.activation = activation(negative_slope=0.3)

        #self.BN = nn.BatchNorm1d(num_features=1, momentum=0.01)


        self.feat_extract = HuBERTWrapper_extractor()
        self.feat_extract.requires_grad_(False)

        
        self.blstm = nn.LSTM(
            input_size=dim_extractor,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True,
        )
        
        
        self.attenPool1 = PoolAttFF(dim_extractor)
        self.attenPool2 = PoolAttFF(dim_extractor)
        self.attenPool3 = PoolAttFF(dim_extractor)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print("----- IN THE MODEL -----")
        #print(x.shape)
        #out = self.BN(x)
        
        out_feats = self.feat_extract(x).permute(0,2,1)
        #print(out_feats.shape)
        out,_ = self.blstm(out_feats)
        #out = out_feats
        out1 = self.attenPool1(out)
        out1 = self.sigmoid(out1)

        out2 = self.attenPool2(out)
        out2 = self.sigmoid(out2)

        out3 = self.attenPool3(out)
        out3 = self.sigmoid(out3)
        #print("----- LEAVING THE MODEL -----")

        return torch.stack([out1,out2,out3],).permute(1,0,2).squeeze()

class HuBERTMetricPredictorFullMulti(nn.Module):


    def __init__(
        self, dim_extractor=768, hidden_size=768//2, activation=nn.LeakyReLU,
    ):
        super().__init__()

        self.activation = activation(negative_slope=0.3)

        #self.BN = nn.BatchNorm1d(num_features=1, momentum=0.01)


        self.feat_extract = HuBERTWrapper_full()
        self.feat_extract.requires_grad_(False)

        
        self.blstm = nn.LSTM(
            input_size=dim_extractor,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True,
        )
        
        
        self.attenPool1 = PoolAttFF(dim_extractor)
        self.attenPool2 = PoolAttFF(dim_extractor)
        self.attenPool3 = PoolAttFF(dim_extractor)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print("----- IN THE MODEL -----")
        #print(x.shape)
        #out = self.BN(x)
        
        out_feats = self.feat_extract(x)
        #print(out_feats.shape)
        out,_ = self.blstm(out_feats)
        #out = out_feats
        out1 = self.attenPool1(out)
        out1 = self.sigmoid(out1)

        out2 = self.attenPool2(out)
        out2 = self.sigmoid(out2)

        out3 = self.attenPool3(out)
        out3 = self.sigmoid(out3)
        #print("----- LEAVING THE MODEL -----")

        return torch.stack([out1,out2,out3],).permute(1,0,2).squeeze()
    
class HuBERTMetricPredictorFullLayersMulti(nn.Module):
    def __init__(
        self, dim_extractor=768, hidden_size=768//2, activation=nn.LeakyReLU,
    ):
        super().__init__()

        self.activation = activation(negative_slope=0.3)


        self.feat_extract = HuBERTWrapper_full_all()
        self.feat_extract.requires_grad_(False)

        
        self.blstm = nn.LSTM(
            input_size=dim_extractor,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True,
        )

        self.layer_weights = nn.Parameter(torch.ones(13))
        self.softmax = nn.Softmax(dim=0)

        self.attenPool1 = PoolAttFF(dim_extractor)
        self.attenPool2 = PoolAttFF(dim_extractor)
        self.attenPool3 = PoolAttFF(dim_extractor)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        #print("----- IN THE MODEL -----")
        #print(x.shape)
        #out = self.BN(x)
        
        out_feats = self.feat_extract(x)
        # out_feats is a list of 13 tensors with each having shape (batch_size,time,768)
        out_feats = torch.stack(out_feats,dim=-1)
        #print(out_feats.shape)
        # out_feats is now of shape (batch_size,time,768,13)
        out_feats = out_feats @ self.softmax(self.layer_weights)
        print(self.layer_weights)
        # out_feats is now of shape (batch_size,time,768)
        #print(out_feats.shape)


        out,_ = self.blstm(out_feats)
        #out = out_feats
        out1 = self.attenPool1(out)
        out1 = self.sigmoid(out1)

        out2 = self.attenPool2(out)
        out2 = self.sigmoid(out2)

        out3 = self.attenPool3(out)
        out3 = self.sigmoid(out3)
        #print("----- LEAVING THE MODEL -----")

        return torch.stack([out1,out2,out3],).permute(1,0,2).squeeze()







