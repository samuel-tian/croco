import torch
from .alligator import AlligatorNet

def args_from_ckpt(ckpt):
    if 'args' in ckpt and hasattr(ckpt['args'], 'model'): # pretrained using the official code release
        s = ckpt['args'].model # eg "CroCoNet(enc_embed_dim=1024, enc_num_heads=16, enc_depth=24)"
        return eval('dict'+s[len('CroCoNet'):]) # transform it into the string of a dictionary and evaluate it
    else: # CroCo v1 released models
        return dict()

class AlligatorDownstreamMonocularEncoder(AlligatorNet):
    def __init__(self,
                 head,
                 **kwargs):
        """ Build network for monocular downstream task, only using the encoder.
        It takes an extra argument head, that is called with the features 
          and a dictionary img_info containing 'width' and 'height' keys
        The head is setup with the croconet arguments in this init function
        NOTE: It works by *calling super().__init__() but with redefined setters
        
        """
        super(AlligatorDownstreamMonocularEncoder, self).__init__(**kwargs)
        head.setup(self)
        self.head = head

    def _set_mask_generator(self, *args, **kwargs):
        """ No mask generator """
        return

    def _set_mask_token(self, *args, **kwargs):
        """ No mask token """
        self.mask_token = None
        return

    def _set_decoder(self, *args, **kwargs):
        """ No decoder """
        return

    def _set_prediction_head(self, *args, **kwargs):
        """ No 'prediction head' for downstream tasks."""
        return

    def forward(self, img):
        """
        img if of size batch_size x 3 x h x w
        """
        B, C, H, W = img.size()
        img_info = {'height': H, 'width': W}
        need_all_layers = hasattr(self.head, 'return_all_blocks') and self.head.return_all_blocks
        out, _, _ = self._encode_image(img, do_mask=False, return_all_blocks=need_all_layers)
        return self.head(out, img_info)
        
        