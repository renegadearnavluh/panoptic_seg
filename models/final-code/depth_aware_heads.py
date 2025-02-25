import torch
import torch.nn.functional as F
import torch.nn as nn
from detectron2.layers import Conv2d, get_norm
from .deform_conv_with_off import ModulatedDeformConvWithOff


def extract_topk_centers(center_map, top_k=10):
    """
    Extracts the top-k highest confidence object centers from the PositionHead output.

    Args:
        center_map (Tensor): Heatmap from PositionHead of shape (B, C, H, W).
        top_k (int): Number of object centers to extract.

    Returns:
        List of (B, top_k, 2) tensors containing (x, y) coordinates of object centers.
    """
    B, C, H, W = center_map.shape
    center_map = center_map.sigmoid()  # Apply sigmoid activation for probability interpretation
    
    # Flatten and extract top-k indices per batch
    center_map_flat = center_map.view(B, C, -1)  # Shape: (B, C, H*W)
    topk_vals, topk_idxs = torch.topk(center_map_flat, top_k, dim=-1)  # Get top-k confidence points
    
    # Convert 1D indices to 2D coordinates
    topk_x = (topk_idxs % W).float()  # Column index (x)
    topk_y = (topk_idxs // W).float()  # Row index (y)
    
    return torch.stack([topk_x, topk_y], dim=-1)  # Shape: (B, C, top_k, 2)


def depth_aware_coord_conv_topk(feat, depth, center_map, top_k=10, alpha=1.0, R=10.0):
    """
    Depth-aware CoordConv using object centers from PositionHead.
    
    Args:
        feat (Tensor): Feature map from FPN of shape (B, C, H, W).
        depth (Tensor): Raw depth map from Cityscapes of shape (B, 1, H_full, W_full).
        center_map (Tensor): Object center heatmap from PositionHead (B, C, H, W).
        top_k (int): Number of object centers to consider per image.
        alpha (float): Scaling factor for depth difference computation.
        R (float): Scaling factor for relative coordinates.

    Returns:
        Tensor: Feature map concatenated with depth-aware CoordConv features.
    """
    B, C, H, W = feat.shape

    # Resize depth map to match feature map size
    depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=False).squeeze(1)  # (B, H, W)

    # Extract top-k object centers from PositionHead
    object_centers = extract_topk_centers(center_map, top_k=top_k)  # (B, C, top_k, 2)

    # Create empty tensors to store feature maps
    Xrel_maps = torch.zeros(B, top_k, H, W, device=feat.device)
    Yrel_maps = torch.zeros(B, top_k, H, W, device=feat.device)
    Ddist_maps = torch.zeros(B, top_k, H, W, device=feat.device)
    F2_5D_maps = torch.zeros(B, top_k, H, W, device=feat.device)

    for b in range(B):  # Iterate over batch
        for k in range(top_k):  # Iterate over top-k centers
            cx, cy = object_centers[b, 0, k]  # Center coordinates (x, y)
            cx, cy = int(cx.item()), int(cy.item())

            # Compute relative coordinates
            Xrel = (torch.arange(H, device=feat.device).float() - cx) / R
            Yrel = (torch.arange(W, device=feat.device).float() - cy) / R
            Xrel, Yrel = torch.meshgrid(Xrel, Yrel, indexing='ij')

            # Compute depth distance map
            Ddist = alpha * (depth[b] - depth[b, cx, cy])  # Depth difference from center point

            # Compute 2.5D distance map
            F2_5D = torch.sqrt(Xrel**2 + Yrel**2 + Ddist**2)

            # Store feature maps
            Xrel_maps[b, k] = Xrel
            Yrel_maps[b, k] = Yrel
            Ddist_maps[b, k] = Ddist
            F2_5D_maps[b, k] = F2_5D

    # Aggregate the maps by taking the **maximum response across all centers**
    Xrel_final = Xrel_maps.max(dim=1).values.clamp(-1, 1)
    Yrel_final = Yrel_maps.max(dim=1).values.clamp(-1, 1)
    Ddist_final = Ddist_maps.max(dim=1).values.clamp(-1, 1)
    F2_5D_final = F2_5D_maps.max(dim=1).values.clamp(-1, 1)

    # Concatenate these features with the original feature map
    extra_features = torch.stack([Xrel_final, Yrel_final, Ddist_final, F2_5D_final], dim=1)  # (B, 4, H, W)
    feat = torch.cat([feat, extra_features], dim=1)  # (B, C+4, H, W)
    
    return feat


def depth_aware_coord_conv(feat, depth, center_map, alpha=1.0, R=10.0):
    """
    Optimized Depth-aware CoordConv with proper depth handling and broadcasting.

    Args:
        feat (Tensor): Feature map from FPN of shape (B, C, H, W).
        depth (Tensor): Raw depth map from Cityscapes of shape (B, H_full, W_full).
        center_map (Tensor): Object center heatmap from PositionHead (B, C, H, W).
        alpha (float): Scaling factor for depth difference computation.
        R (float): Scaling factor for relative coordinates.

    Returns:
        Tensor: Feature map concatenated with depth-aware CoordConv features.
    """
    B, C, H, W = feat.shape
    
    # Normalize depth to [0,1] and resize
    depth = depth / 255.0  # Normalize
    depth_resized = F.interpolate(depth.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)  # (B, H, W)

    # Create coordinate grids with correct shape (H, W)
    y_pos = torch.linspace(-1, 1, H, device=feat.device)
    x_pos = torch.linspace(-1, 1, W, device=feat.device)
    grid_y, grid_x = torch.meshgrid(y_pos, x_pos, indexing="ij")  # Ensure correct shape (H, W)
    
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)

    # Sigmoid activation for center map
    center_map = center_map.sigmoid()  # (B, C, H, W)

    # Flatten center_map across spatial dimensions (H * W) and find max confidence
    center_map_flat = center_map.view(B, C, -1)  # (B, C, H*W)
    top_vals, top_idx = torch.max(center_map_flat, dim=-1)  # (B, C)

    # Select the highest confidence class per batch
    best_class = top_vals.max(dim=-1).indices  # (B,)
    idx = top_idx[torch.arange(B), best_class]  # (B,)

    # Compute (cx, cy) coordinates of the center
    cx = (idx % W).view(B, 1, 1).float()  # (B, 1, 1)
    cy = (idx // W).view(B, 1, 1).float()  # (B, 1, 1)

    # Generate coordinate grids
    y_coords = torch.arange(H, device=feat.device).view(1, H, 1).expand(B, -1, W)  # (B, H, W)
    x_coords = torch.arange(W, device=feat.device).view(1, 1, W).expand(B, H, -1)  # (B, H, W)

    # Compute relative coordinates
    Xrel = (x_coords - cx) / R  # (B, H, W)
    Yrel = (y_coords - cy) / R  # (B, H, W)

    # Convert center coordinates to indices for depth gathering
    cx_idx = cx.long().clamp(0, W - 1)
    cy_idx = cy.long().clamp(0, H - 1)

    # Gather depth at (cx, cy) locations
    D_center = depth_resized[torch.arange(B), cy_idx.squeeze(), cx_idx.squeeze()].view(B, 1, 1)  # (B, 1, 1)

    # Broadcast D_center across (H, W)
    D_center = D_center.expand(B, H, W)  # (B, H, W)

    # Compute depth difference
    Ddist = alpha * (depth_resized - D_center)  # (B, H, W)

    # Compute 2.5D distance map
    F2_5D = torch.sqrt(Xrel**2 + Yrel**2 + Ddist**2)  # (B, H, W)

    # Stack and clamp feature maps
    extra_features = torch.stack([grid_x, grid_y, Xrel, Yrel, Ddist, F2_5D], dim=1).clamp(-1, 1)  # (B, 6, H, W)

    # Concatenate with original feature map
    feat = torch.cat([feat, extra_features], dim=1)  # (B, C+6, H, W)

    return feat

class SingleHead(nn.Module):
    """
    Build single head with convolutions and coord conv.
    """
    def __init__(self, in_channel, conv_dims, num_convs, deform=False, topk=False, norm='', name='', alpha=1.0, R=10.0):
        super().__init__()
        self.topk = topk
        self.conv_norm_relus = []
        if deform:
            conv_module = ModulatedDeformConvWithOff
        else:
            conv_module = Conv2d
        for k in range(num_convs):
            conv = conv_module(
                    in_channel if k==0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=get_norm(norm, conv_dims),
                    activation=F.relu,
                )
            self.add_module("{}_head_{}".format(name, k + 1), conv)
            self.conv_norm_relus.append(conv)
            self.alpha = alpha
            self.R = R

    def forward(self, x, depth, center_map):
        if self.topk:
            x = depth_aware_coord_conv_topk(x, depth, center_map, alpha=self.alpha, R=self.R)
        else:
            x = depth_aware_coord_conv(x, depth, center_map, alpha=self.alpha, R=self.R)
        for layer in self.conv_norm_relus:
            x = layer(x)
        return x
    
class DepthAwareKernelHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_channel      = cfg.MODEL.FPN.OUT_CHANNELS
        conv_dims       = cfg.MODEL.KERNEL_HEAD.CONVS_DIM
        num_convs       = cfg.MODEL.KERNEL_HEAD.NUM_CONVS
        deform          = cfg.MODEL.KERNEL_HEAD.DEFORM
        topk           = False
        norm            = cfg.MODEL.KERNEL_HEAD.NORM

        self.kernel_head = SingleHead(in_channel+6 , 
                                      conv_dims,
                                      num_convs,
                                      deform=deform,
                                      topk=topk,
                                      norm=norm,
                                      name='kernel_head')
        self.out_conv = Conv2d(conv_dims, conv_dims, kernel_size=3, padding=1)
        nn.init.normal_(self.out_conv.weight, mean=0, std=0.01)
        if self.out_conv.bias is not None:
            nn.init.constant_(self.out_conv.bias, 0)
       
    def forward(self, feat, depth, center_map):
        x = self.kernel_head(feat, depth, center_map)
        x = self.out_conv(x)
        return x
    
def build_depth_aware_kernel_head(cfg):
    return DepthAwareKernelHead(cfg)