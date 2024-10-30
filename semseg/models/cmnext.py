import torch
from torch import Tensor
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads import SegFormerHead
from semseg.models.heads import LightHamHead
from semseg.models.heads import UPerHead
from fvcore.nn import flop_count_table, FlopCountAnalysis


class CMNeXt(BaseModel):
    def __init__(self, weight_h_ori, backbone: str = 'CMNeXt-B0', num_classes: int = 25,
                 modals: list = ['img', 'depth', 'event', 'lidar']) -> None:
        super().__init__(weight_h_ori, backbone, num_classes, modals)
        # print('base model cmnext weight_h_ori', weight_h_ori)
        self.decode_head_f = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 512,
                                           num_classes)
        self.decode_head_d = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 512,
                                           num_classes)
        self.decode_head_e = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 512,
                                           num_classes)
        self.decode_head_l = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 512,
                                           num_classes)
        self.apply(self._init_weights)

    def forward(self, x: list) -> list:
        # print('base model cmnext forward input', x[0].shape)
        y_f, y_ext = self.backbone(x)
        y_f = self.decode_head_f(y_f)
        y_f = F.interpolate(y_f, size=x[0].shape[2:], mode='bilinear', align_corners=False)
        if self.training:
            print('training!!!')
            y_d = self.decode_head_d(y_ext[0])
            y_e = self.decode_head_e(y_ext[1])
            y_l = self.decode_head_l(y_ext[2])

            y_d = F.interpolate(y_d, size=x[0].shape[2:], mode='bilinear', align_corners=False)
            y_e = F.interpolate(y_e, size=x[0].shape[2:], mode='bilinear', align_corners=False)
            y_l = F.interpolate(y_l, size=x[0].shape[2:], mode='bilinear', align_corners=False)
            return y_f, y_d, y_e, y_l
        else:
            return y_f

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            if self.backbone.num_modals > 0:
                load_dualpath_model(self.backbone, pretrained)
            else:
                checkpoint = torch.load(pretrained, map_location='cpu')
                if 'state_dict' in checkpoint.keys():
                    checkpoint = checkpoint['state_dict']
                if 'model' in checkpoint.keys():
                    checkpoint = checkpoint['model']
                msg = self.backbone.load_state_dict(checkpoint, strict=False)
                print(msg)


def load_dualpath_model(model, model_file):
    extra_pretrained = None
    if isinstance(extra_pretrained, str):
        raw_state_dict_ext = torch.load(extra_pretrained, map_location=torch.device('cpu'))
        if 'state_dict' in raw_state_dict_ext.keys():
            raw_state_dict_ext = raw_state_dict_ext['state_dict']
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file

    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('patch_embed') >= 0:
            state_dict[k] = v
        elif k.find('block') >= 0:
            state_dict[k] = v
            # state_dict[k.replace('block', 'shared_extra_block')] = v
            # state_dict[k.replace('block', 'diff1_extra_block')] = v
            # state_dict[k.replace('block', 'diff2_extra_block')] = v
            # state_dict[k.replace('block', 'diff3_extra_block')] = v
        elif k.find('norm') >= 0:
            state_dict[k] = v
            # state_dict[k.replace('norm', 'shared_extra_norm')] = v
            # state_dict[k.replace('norm', 'diff1_extra_norm')] = v
            # state_dict[k.replace('norm', 'diff2_extra_norm')] = v
            # state_dict[k.replace('norm', 'diff3_extra_norm')] = v

    if isinstance(extra_pretrained, str):
        for k, v in raw_state_dict_ext.items():
            if k.find('patch_embed1.proj') >= 0:
                state_dict[k.replace('patch_embed1.proj', 'extra_downsample_layers.0.proj.module')] = v
            if k.find('patch_embed2.proj') >= 0:
                state_dict[k.replace('patch_embed2.proj', 'extra_downsample_layers.1.proj.module')] = v
            if k.find('patch_embed3.proj') >= 0:
                state_dict[k.replace('patch_embed3.proj', 'extra_downsample_layers.2.proj.module')] = v
            if k.find('patch_embed4.proj') >= 0:
                state_dict[k.replace('patch_embed4.proj', 'extra_downsample_layers.3.proj.module')] = v

            if k.find('patch_embed1.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed1.norm', 'extra_downsample_layers.0.norm.ln_{}'.format(i))] = v
            if k.find('patch_embed2.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed2.norm', 'extra_downsample_layers.1.norm.ln_{}'.format(i))] = v
            if k.find('patch_embed3.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed3.norm', 'extra_downsample_layers.2.norm.ln_{}'.format(i))] = v
            if k.find('patch_embed4.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed4.norm', 'extra_downsample_layers.3.norm.ln_{}'.format(i))] = v
            elif k.find('block') >= 0:
                state_dict[k.replace('block', 'extra_block')] = v
            elif k.find('norm') >= 0:
                state_dict[k.replace('norm', 'extra_norm')] = v

    msg = model.load_state_dict(state_dict, strict=False)
    del state_dict


if __name__ == '__main__':
    device = torch.device('cuda')
    modals = ['img', 'depth', 'event', 'lidar']
    x = [torch.zeros(2, 3, 1024, 1024).to(device), torch.ones(2, 3, 1024, 1024).to(device), (torch.ones(2, 3, 1024, 1024) * 2).to(device),
         (torch.ones(2, 3, 1024, 1024) * 3).to(device)]
    model = CMNeXt(int(x[0].shape[2] / 4), 'CMNeXt-B0', 25, modals).to(device)
    model.init_pretrained('/home/yi/Documents/DELIVER/checkpoints/pretrained/segformer/mit_b0.pth')
    model.eval()
    y1= model(x)
    print(len(y1))
    # print(moe_loss)
