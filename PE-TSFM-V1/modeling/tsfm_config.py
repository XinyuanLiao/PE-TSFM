from transformers import PretrainedConfig

class TSFMConfig(PretrainedConfig):
    model_type = "tsfm"

    def __init__(self, 
                 patch_size=16,
                 context_length=512,
                 input_channels=5,
                 dim=256,
                 dim_per_heads=64,
                 num_layers=4,
                 ffn_ratio=4.0,
                 norm_eps: float = 1e-8,
                 dropout=0.2,
                 head_dropout=0.2,
                 mask_ratio=0.2,
                 num_classes=4,
                 label_smoothing=0.1,
                 init_std=0.02,
                 mask_token_trainable=False,
                 shared_embedding=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.context_length = context_length
        self.input_channels = input_channels
        self.dim = dim
        self.dim_per_heads = dim_per_heads
        self.num_layers = num_layers
        self.ffn_ratio = ffn_ratio
        self.norm_eps = norm_eps
        self.dropout = dropout
        self.head_dropout = head_dropout
        self.mask_ratio = mask_ratio
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.init_std = init_std
        self.mask_token_trainable = mask_token_trainable
        self.shared_embedding = shared_embedding