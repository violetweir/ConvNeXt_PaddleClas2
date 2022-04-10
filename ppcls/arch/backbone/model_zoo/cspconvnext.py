from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .convnext import Block,LayerNorm, drop_path, ClasHead, _load_pretrained
import paddle 
import paddle.fluid as fluid
import paddle.nn as nn
from paddle import ParamAttr

from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url


trunc_normal_ = nn.initializer.TruncatedNormal(std=0.02)
zeros_ = nn.initializer.Constant(value=0.0)
ones_ = nn.initializer.Constant(value=1.0)

MODEL_URLS = {
    "ConvNext_tiny":
    "https://passl.bj.bcebos.com/models/convnext_tiny_1k_224.pdparams",
    "ConvNext_small":
    "https://passl.bj.bcebos.com/models/convnext_small_1k_224.pdparams",
}

class L2Decay(fluid.regularizer.L2Decay):
    def __init__(self, coeff=0.0):
        super(L2Decay, self).__init__(coeff)

class EffectiveSELayer(nn.Layer):
    """ Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channels, act='hardsigmoid'):
        super(EffectiveSELayer, self).__init__()
        self.fc = nn.Conv2D(channels, channels, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)



class ConvBNLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False)

        self.bn = nn.BatchNorm2D(
            ch_out,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class CSPStage(nn.Layer):
    def __init__(self,
                block_fn,
                ch_in,
                ch_out,
                n,
                stride,
                p_rates,
                layer_scale_init_value=1e-6,
                act=nn.GELU,
                attn='eca'):
        super().__init__()
        ch_mid = (ch_in+ch_out)//2
        if stride == 2:
            self.down = nn.Sequential(
                nn.Conv2D(ch_in, ch_mid, kernel_size=3, stride=2, padding=1),
                LayerNorm(ch_mid, epsilon=1e-6, data_format="channels_first"),
                nn.GELU()
            )
        else:
            self.down = None
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.blocks = nn.Sequential(*[
            block_fn(
                ch_mid // 2, drop_path=p_rates[i],layer_scale_init_value=layer_scale_init_value)
            for i in range(n)
        ])
        if attn:
            self.attn = EffectiveSELayer(ch_mid, act='hardsigmoid')
        else:
            self.attn = None

        self.conv3 = ConvBNLayer(ch_mid, ch_out, 1, act=act)

    def forward(self, x):
        if self.down is not None:
            x = self.down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = paddle.concat([y1, y2], axis=1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y

class CSPConvNext(nn.Layer):
    def __init__(
        self,
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[96, 96, 192, 384, 768],
        drop_path_rate=0.,
        layer_scale_init_value=1e-6,
        stride=[1,2,2,2],
        return_idx=[1,2,3]
    ):
        super().__init__()
        self.Down_Conv = nn.Sequential(
            nn.Conv2D(3, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], epsilon=1e-6, data_format="channels_first"))
        dp_rates = [
            x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))
        ]
        n = len(depths)

        self.stages = nn.Sequential(*[(str(i), CSPStage(
            Block, dims[i], dims[i + 1], depths[i], stride[i], dp_rates[sum(depths[:i]) : sum(depths[:i+1])],act=nn.GELU))
                                      for i in range(n)])
        self.norm = nn.LayerNorm(dims[-1], epsilon=1e-6) 

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2D, nn.Linear)):
            trunc_normal_(m.weight)
            zeros_(m.bias)
    
    
    def forward(self, inputs):
        x = inputs
        x = self.Down_Conv(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            # if idx in self.return_idx:
            #     outs.append(x)
        return self.norm(x.mean([-2, -1]))

class Model(paddle.nn.Layer):
    def __init__(self, class_num=1000,):
        super().__init__()
        self.backbone = CSPConvNext()
        self.head = ClasHead(with_avg_pool=False, in_channels=768, num_classes=class_num)

    def forward(self, x):

        x = self.backbone(x)
        x = self.head(x)
        return x

def CSPConvNext_tiny(pretrained=False, use_ssld=False, **kwargs):
    model = Model( **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ConvNext_tiny"], use_ssld=use_ssld)
    return model

if __name__=="__main__":
     model  = CSPConvNext_tiny()
     paddle.summary(model,(1,3,224,224))





        
        