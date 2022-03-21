import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DAMWrapper(torch.nn.Module):

    def __init__(self, tensor_size, tau):
        super(DAMWrapper, self).__init__()
        self.tensor_size = tensor_size if isinstance(tensor_size, list) else [tensor_size]

        # get head num
        if len(tensor_size) == 3:  # [head_num, N, N] where N indicates the token lenth
            self.num_attention_heads = tensor_size[0]

        # get token lenth
        self.N = tensor_size[-1]
        
        # init the alphas
        self.init_alphas = torch.nn.Parameter(1e-3 * torch.randn(self.num_attention_heads, self.N, 2).to(device))
        self.tau = tau

    def forward(self):
        # Initialize the mask using gumbel sigmoid
        eps = 1e-5
        gumbels = -(torch.empty_like(self.init_alphas).exponential_() + eps).log()  # ~Gumbel(0,1)
        gumbels = (self.init_alphas + gumbels) / self.tau  # ~Gumbel(logits,tau)
        mask = gumbels.softmax(-1)

        # Generate differentiabal attention masks (DAM) with structured constrain
        mask = mask[:, :, 0].unsqueeze(2)   # (num_heads, N, 1)
        mask = mask.expand(self.num_attention_heads, self.N, self.N)
        mask = mask.triu()
        mask = torch.rot90(mask, 1, (1, 2))
        mask_tri = torch.zeros(self.num_attention_heads, self.N, self.N).to(device)
        mask_tri[:, 0] = mask[:, 0]
        for i in range(1, self.N):
            mask_tri[:, i, i:] = mask[:, i, :-i]
        masks = mask_tri + torch.transpose(torch.triu(mask_tri, 1), 1, 2)

        # For capability
        masks = masks.to(dtype=next(self.parameters()).dtype)

        mask_normalize = (1.0 - masks) * -10000.0

        return [mask_normalize, masks]
