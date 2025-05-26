import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FlowHead2(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead2, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 10*self.tanh(self.conv2(self.relu(self.conv1(x))))


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h


class LiteUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=32, input_dim=16, num_outputs=1, downsample=8):
        super(LiteUpdateBlock, self).__init__()
        self.upsample_mask_dim = downsample * downsample
        self.num_outputs = num_outputs
        assert self.num_outputs in [1, 2]
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=input_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=hidden_dim)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.upsample_mask_dim*9, 1, padding=0))
        if self.num_outputs == 2:
            self.flow_head2 = FlowHead(hidden_dim, hidden_dim=hidden_dim)
            self.mask2 = nn.Sequential(
                nn.Conv2d(hidden_dim, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, self.upsample_mask_dim*9, 1, padding=0))

    def forward(self, net, inp):
        return self.gru(net, inp)

    def compute_deltaflow(self, net):
        return self.flow_head(net)

    def compute_nextflow(self, net):
        if self.num_outputs == 2:
            return self.flow_head2(net)
        else:
            raise NotImplementedError

    def compute_up_mask(self, net):
        return self.mask(net)

    def compute_up_mask2(self, net):
        if self.num_outputs == 2:
            return self.mask2(net)
        else:
            raise NotImplementedError


class SelfAttention(nn.Module):
    """Self-attention module for feature refinement."""
    def __init__(self, channels, reduction=8):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Reshape for attention computation
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, height * width)
        attention = self.softmax(torch.bmm(proj_query, proj_key))
        
        proj_value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        return out


class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extraction module."""
    def __init__(self, input_dim, output_dim):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.conv1x1 = nn.Conv2d(input_dim, output_dim // 4, kernel_size=1)
        self.conv3x3 = nn.Conv2d(input_dim, output_dim // 4, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(input_dim, output_dim // 4, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(input_dim, output_dim // 4, kernel_size=7, padding=3)
        self.norm = nn.GroupNorm(4, output_dim)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x1 = self.conv1x1(x)
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        x7 = self.conv7x7(x)
        out = torch.cat([x1, x3, x5, x7], dim=1)
        out = self.relu(self.norm(out))
        return out


class MultiScaleAttentionGRU(nn.Module):
    """Enhanced GRU with multi-scale feature processing and attention."""
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(MultiScaleAttentionGRU, self).__init__()
        
        # Multi-scale feature processing
        self.input_feature_extractor = MultiScaleFeatureExtractor(input_dim, input_dim)
        self.hidden_feature_extractor = MultiScaleFeatureExtractor(hidden_dim, hidden_dim)
        
        # GRU components with residual connections
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        
        # Attention mechanism
        self.attention = SelfAttention(hidden_dim)
        
        # Residual connection
        self.res_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        
    def forward(self, h, x):
        # Apply multi-scale feature extraction
        x_ms = self.input_feature_extractor(x)
        h_ms = self.hidden_feature_extractor(h)
        
        # Original GRU operations
        hx = torch.cat([h_ms, x_ms], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h_ms, x_ms], dim=1)))
        
        # Update state
        h_new = (1-z) * h_ms + z * q
        
        # Apply attention and residual connection
        h_att = self.attention(h_new)
        h_res = h_att + self.res_conv(h)
        
        return h_res


class EnhancedFlowHead(nn.Module):
    """Enhanced flow prediction head with uncertainty estimation."""
    def __init__(self, input_dim=128, hidden_dim=256, with_uncertainty=False):
        super(EnhancedFlowHead, self).__init__()
        self.with_uncertainty = with_uncertainty
        output_channels = 3 if with_uncertainty else 2
        
        # Multi-level features
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim // 2, output_channels, 3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.GroupNorm(8, hidden_dim)
        self.norm2 = nn.GroupNorm(4, hidden_dim // 2)
        
    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.conv3(x)
        
        if self.with_uncertainty:
            # Split output into flow and uncertainty
            flow = x[:, :2]
            uncertainty = torch.sigmoid(x[:, 2:3])  # Sigmoid to bound uncertainty between 0 and 1
            return flow, uncertainty
        else:
            return x


class EnhancedUpdateBlock(nn.Module):
    """Enhanced update block with multi-scale processing and attention mechanism."""
    def __init__(self, 
                 hidden_dim=32, 
                 input_dim=16, 
                 num_outputs=1, 
                 downsample=8, 
                 with_uncertainty=False,
                 use_attention=True):
        super(EnhancedUpdateBlock, self).__init__()
        self.upsample_mask_dim = downsample * downsample
        self.num_outputs = num_outputs
        self.with_uncertainty = with_uncertainty
        assert self.num_outputs in [1, 2]
        
        # Use enhanced GRU with attention if specified
        if use_attention:
            self.gru = MultiScaleAttentionGRU(hidden_dim=hidden_dim, input_dim=input_dim)
        else:
            self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=input_dim)
        
        # Enhanced flow prediction with uncertainty estimation
        self.flow_head = EnhancedFlowHead(hidden_dim, hidden_dim=hidden_dim, with_uncertainty=with_uncertainty)
        
        # Enhanced upsampling mask - still using the same interface for compatibility
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.upsample_mask_dim*9, 1, padding=0))
        
        if self.num_outputs == 2:
            self.flow_head2 = EnhancedFlowHead(hidden_dim, hidden_dim=hidden_dim, with_uncertainty=with_uncertainty)
            self.mask2 = nn.Sequential(
                nn.Conv2d(hidden_dim, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, self.upsample_mask_dim*9, 1, padding=0))
                
        # Store uncertainties for later use if needed
        self.flow_uncertainty = None
        self.flow2_uncertainty = None
    
    def forward(self, net, inp):
        """Forward pass through the GRU."""
        return self.gru(net, inp)
    
    def compute_deltaflow(self, net):
        """Compute delta flow, with optional uncertainty."""
        result = self.flow_head(net)
        if self.with_uncertainty:
            flow, uncertainty = result
            # Store uncertainty for potential later use
            self.flow_uncertainty = uncertainty
            return flow
        else:
            return result
    
    def compute_nextflow(self, net):
        """Compute next flow prediction, with optional uncertainty."""
        if self.num_outputs == 2:
            result = self.flow_head2(net)
            if self.with_uncertainty:
                flow, uncertainty = result
                # Store uncertainty for potential later use
                self.flow2_uncertainty = uncertainty
                return flow
            else:
                return result
        else:
            raise NotImplementedError
    
    def compute_up_mask(self, net):
        """Compute upsampling mask for the flow."""
        return self.mask(net)
    
    def compute_up_mask2(self, net):
        """Compute upsampling mask for the next flow."""
        if self.num_outputs == 2:
            return self.mask2(net)
        else:
            raise NotImplementedError
            
    def get_flow_uncertainty(self):
        """Retrieve stored flow uncertainty."""
        return self.flow_uncertainty
        
    def get_nextflow_uncertainty(self):
        """Retrieve stored next flow uncertainty."""
        return self.flow2_uncertainty
