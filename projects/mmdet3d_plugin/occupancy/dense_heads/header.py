import torch.nn as nn

                                                        
class Header(nn.Module):
    def __init__(
        self,
        class_num,
        feature,
    ):
        super(Header, self).__init__()
        self.feature = feature
        self.class_num = class_num
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.feature),
            nn.Linear(self.feature, self.class_num),
        )

        self.up_scale_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)


    def forward(self, x3d_l1):
        # [1, 64, 128, 128, 16]
        # res = {} 

        x3d_up_l1 = self.up_scale_2(x3d_l1) # [1, dim, 128, 128, 16] -> [1, dim, 256, 256, 32]

        bs, feat_dim, w, l, h  = x3d_up_l1.shape

        x3d_up_l1 = x3d_up_l1.permute(0,2,3,4,1).reshape(bs,-1, feat_dim)

        ssc_logit_full = self.mlp_head(x3d_up_l1)

        res= ssc_logit_full.reshape(bs, w, l, h, self.class_num).permute(0,4,1,2,3)

        return res
