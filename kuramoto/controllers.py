import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

# =========================
# Circular mean phase psi(t)
# =========================
def global_order_and_mean_phase(theta: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute the global order parameter magnitude and the circular mean phase.

    Args:
        theta: Tensor of shape (N,) or (1, N).

    Returns:
        Dictionary with:
            r: scalar tensor with the order parameter magnitude.
            psi: scalar tensor in [-pi, pi] with the mean phase.
    """
    if theta.dim() == 1:
        theta = theta.view(1, -1)
    z = torch.exp(1j * theta)        # Complex phasors associated with each oscillator, shape (1, N)
    zbar = torch.mean(z, dim=1)      # Mean complex order parameter, shape (1,)
    r = torch.abs(zbar).real.squeeze(0)
    psi = torch.angle(zbar).squeeze(0)
    return {"r": r, "psi": psi}

# =========================
# Baseline: mean-phase feedback
# =========================
def baseline_mean_phase_feedback(
    theta: torch.Tensor,
    beta: float,
    gating: bool = False,
    alpha: float = 10.0
) -> torch.Tensor:
    """
    Compute the baseline mean-phase feedback controller.

    Control law:
        u_i(t) = -beta * sin(theta_i - psi)

    Optionally, the control is modulated by tanh(alpha * (1 - r)).

    Args:
        theta: Tensor of shape (1, N) or (N,).
        beta: Consensus gain.
        gating: Whether to apply the global gating factor.
        alpha: Gating sharpness parameter.

    Returns:
        Control tensor with the same shape convention as the input.
    """
    squeeze_back = False
    if theta.dim() == 1:
        theta_in = theta.view(1, -1)
        squeeze_back = True
    else:
        theta_in = theta

    info = global_order_and_mean_phase(theta_in)
    r = info["r"]        # Scalar global order parameter magnitude
    psi = info["psi"]    # Scalar circular mean phase

    u = -beta * torch.sin(theta_in - psi.view(1, 1))  # Baseline consensus feedback, shape (1, N)

    if gating:
        gate = torch.tanh(alpha * (1.0 - r)).view(1, 1)  # Global modulation factor based on desynchronization
        u = u * gate

    return (u.squeeze(0), 0) if squeeze_back else (u, 0)

# =========================
# Quick sweep helper
# =========================
def make_beta_sweep(betas):
    """
    Build a list of controller callables for a sweep over beta values.

    Args:
        betas: Iterable of gain values.

    Returns:
        List of callables of the form controller(theta) -> u.
    """
    ctrls = []
    for b in betas:
        ctrls.append(lambda th, bb=b: baseline_mean_phase_feedback(th, bb, gating=True, alpha=10.0))
    return ctrls

class MeanPhaseFeedbackController:
    def __init__(self, beta: float, gating: bool = True, alpha: float = 10.0):
        """
        Wrapper class for the baseline mean-phase feedback controller.

        Args:
            beta: Consensus gain.
            gating: Whether to use global gating.
            alpha: Gating sharpness parameter.
        """
        self.beta = beta
        self.gating = gating
        self.alpha = alpha

    def __call__(self, theta_measured: torch.Tensor) -> torch.Tensor:
        """
        Apply the baseline controller to the measured phase vector.

        Args:
            theta_measured: Tensor of shape (1, N) or (N,).

        Returns:
            Control tensor and auxiliary output from the baseline controller.
        """
        # theta_measured: (1, N) or (N,)
        return baseline_mean_phase_feedback(
            theta_measured, beta=self.beta, gating=self.gating, alpha=self.alpha
        )
    

# =========================
# Neural-Resilient Pinning Controller
# =========================

class NRPConsensusController:
    def __init__(
        self,
        neural_model,
        beta: float,
        gating: bool = True,
        alpha: float = 10.0,
        use_sigmoid: bool = True,
        threshold: float = 0.55,
        k_select: Optional[int] = None,
        u_clip: Optional[float] = None
    ):
        """
        Neural Resilient Pinning (NRP) controller with discrete node selection.

        Args:
            neural_model: Model that outputs logits z with the same size as theta.
            beta: Consensus gain.
            gating: Whether to use the global gate tanh(alpha * (1 - r)).
            alpha: Gate parameter.
            use_sigmoid: Whether to apply a sigmoid before thresholding or top-k selection.
            threshold: Threshold used when top-k selection is not applied.
            k_select: If not None, activate exactly k nodes using top-k selection.
            u_clip: Optional saturation bound for the control input.
        """
        self.model = neural_model
        self.beta = beta
        self.gating = gating
        self.alpha = alpha
        self.use_sigmoid = use_sigmoid
        self.threshold = threshold
        self.k_select = k_select
        self.u_clip = u_clip

    def __call__(self, theta_measured: torch.Tensor):
        """
        Compute the NRP control action and the binary activation mask.

        Args:
            theta_measured: Tensor of shape (1, N) or (N,).

        Returns:
            Tuple (u, m) with control input and node selection mask.
        """
        squeeze_back = False
        if theta_measured.dim() == 1:
            theta = theta_measured.view(1, -1)
            squeeze_back = True
        else:
            theta = theta_measured

        # 1) Base consensus control
        info = global_order_and_mean_phase(theta)
        r = info["r"]
        psi = info["psi"]

        u_cons = -self.beta * torch.sin(theta - psi.view(1, 1))

        if self.gating:
            gate = torch.tanh(self.alpha * (1.0 - r)).view(1, 1)
            u_cons = u_cons * gate

        # 2) Network scores
        z = self.model(theta_measured)
        if z.dim() == 1:
            z = z.view(1, -1)

        scores = torch.sigmoid(z) if self.use_sigmoid else z
        B, N = scores.shape

        # 3) Selection mask
        if self.k_select is not None:
            k = max(1, min(self.k_select, N))
            m = torch.zeros_like(scores)

            # Top-k selection for each batch element
            topk_idx = torch.topk(scores, k=k, dim=1).indices
            m.scatter_(1, topk_idx, 1.0)
        else:
            if self.use_sigmoid:
                m = (scores > self.threshold).float()
            else:
                m = torch.clamp(scores, 0.0, 1.0)

        # 4) Apply masked control
        u = m * u_cons

        if self.u_clip is not None:
            u = torch.clamp(u, -self.u_clip, self.u_clip)

        return (u.squeeze(0), m.squeeze(0)) if squeeze_back else (u, m)


class FixedKMaskModel(nn.Module):
    def __init__(self, N, selected_indices, high=10.0, low=-10.0):
        """
        Deterministic mask model that always assigns high logits to a fixed set of nodes.

        Args:
            N: Total number of nodes.
            selected_indices: Indices of nodes that should be selected.
            high: Logit value assigned to selected nodes.
            low: Logit value assigned to non-selected nodes.
        """
        super().__init__()
        self.N = N
        self.selected_indices = selected_indices
        self.high = high
        self.low = low

    def forward(self, theta):
        """
        Return fixed logits independently of the input phase vector.

        Args:
            theta: Input tensor used only to infer batch size and device.

        Returns:
            Tensor of logits with shape (B, N).
        """
        if theta.dim() == 1:
            B = 1
            device = theta.device
        else:
            B = theta.shape[0]
            device = theta.device

        z = torch.full((B, self.N), self.low, device=device)
        z[:, self.selected_indices] = self.high
        return z


class RandomKMaskModel(nn.Module):
    def __init__(self, N: int, k: int):
        """
        Random mask model that selects k nodes uniformly at random for each batch element.

        Args:
            N: Total number of nodes.
            k: Number of selected nodes.
        """
        super().__init__()
        self.N = N
        self.k = k

    def forward(self, theta):
        """
        Generate random logits for node selection.

        Args:
            theta: Input tensor used only to infer batch size and device.

        Returns:
            Tensor of logits with shape (B, N).
        """
        if theta.dim() == 1:
            B = 1
            device = theta.device
        else:
            B = theta.shape[0]
            device = theta.device

        z = torch.full((B, self.N), -10.0, device=device)

        for b in range(B):
            idx = torch.randperm(self.N, device=device)[:self.k]
            z[b, idx] = 10.0

        return z


class NRPConsensusControllerTrainable:
    def __init__(
        self,
        neural_model,
        beta: float,
        gating: bool = True,
        alpha: float = 10.0,
        use_sigmoid: bool = True,
        threshold: float = 0.55,
        k_select: Optional[int] = None,
        u_clip: Optional[float] = None,
        soft_selection: bool = False
    ):
        """
        Trainable NRP controller supporting both soft and discrete selection.

        Args:
            neural_model: Model that outputs node logits.
            beta: Consensus gain.
            gating: Whether to use the global gate tanh(alpha * (1 - r)).
            alpha: Gate parameter.
            use_sigmoid: Whether to apply a sigmoid to logits.
            threshold: Threshold used in discrete threshold-based selection.
            k_select: If not None, activate exactly k nodes using top-k selection.
            u_clip: Optional saturation bound for the control input.
            soft_selection:
                True  -> differentiable training mode, use m = sigmoid(z).
                False -> discrete evaluation mode (top-k or threshold).
        """
        self.model = neural_model
        self.beta = beta
        self.gating = gating
        self.alpha = alpha
        self.use_sigmoid = use_sigmoid
        self.threshold = threshold
        self.k_select = k_select
        self.u_clip = u_clip
        self.soft_selection = soft_selection

    def __call__(self, theta_measured: torch.Tensor):
        """
        Compute the trainable NRP control action and node mask.

        Args:
            theta_measured: Tensor of shape (1, N) or (N,).

        Returns:
            Tuple (u, m) with control input and soft/discrete selection mask.
        """
        squeeze_back = False
        if theta_measured.dim() == 1:
            theta = theta_measured.view(1, -1)
            squeeze_back = True
        else:
            theta = theta_measured

        # 1) Base consensus control
        info = global_order_and_mean_phase(theta)
        r = info["r"]
        psi = info["psi"]

        u_cons = -self.beta * torch.sin(theta - psi.view(1, 1))

        if self.gating:
            gate = torch.tanh(self.alpha * (1.0 - r)).view(1, 1)
            u_cons = u_cons * gate

        # 2) Network scores
        z = self.model(theta_measured)
        if z.dim() == 1:
            z = z.view(1, -1)

        scores = torch.sigmoid(z) if self.use_sigmoid else z
        B, N = scores.shape

        # 3) Selection mask
        if self.soft_selection:
            # Differentiable training mode
            m = scores
        else:
            # Discrete evaluation mode
            if self.k_select is not None:
                k = max(1, min(self.k_select, N))
                m = torch.zeros_like(scores)
                topk_idx = torch.topk(scores, k=k, dim=1).indices
                m.scatter_(1, topk_idx, 1.0)
            else:
                if self.use_sigmoid:
                    m = (scores > self.threshold).float()
                else:
                    m = torch.clamp(scores, 0.0, 1.0)

        # 4) Apply masked control
        u = m * u_cons

        if self.u_clip is not None:
            u = torch.clamp(u, -self.u_clip, self.u_clip)

        return (u.squeeze(0), m.squeeze(0)) if squeeze_back else (u, m)

# Per-node network

class NodeMaskNet(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=128):
        """
        Per-node neural network that maps node features to a scalar logit.

        Args:
            in_dim: Number of input features per node.
            hidden_dim: Hidden layer width.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_features):
        """
        Compute a logit for each node.

        Args:
            node_features: Tensor of shape [B, N, F].

        Returns:
            Tensor of shape [B, N] with one logit per node.
        """
        # node_features: [B, N, F]
        z = self.net(node_features).squeeze(-1)   # Collapse the last dimension to get one score per node
        return z


# Per-node features

def build_node_features(theta, adjacency):
    """
    Build local and global node-wise features from oscillator phases.

    Args:
        theta: Tensor of shape [B, N] or [N].
        adjacency: Adjacency matrix of shape [N, N].

    Returns:
        Feature tensor of shape [B, N, 8].
    """
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)

    B, N = theta.shape
    device = theta.device

    # Global order statistics
    exp_theta = torch.exp(1j * theta)
    mean_complex = torch.mean(exp_theta, dim=1)
    psi = torch.angle(mean_complex)  # Circular mean phase, shape [B]
    psi = psi.view(B, 1)

    # Global phase-related features
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    sin_err = torch.sin(theta - psi)
    cos_err = torch.cos(theta - psi)

    # Neighborhood averages
    A = adjacency.to(device).float()
    deg = A.sum(dim=1, keepdim=True).clamp(min=1.0)  # Node degree, shape [N, 1]

    sin_neigh = (sin_t @ A.T) / deg.T
    cos_neigh = (cos_t @ A.T) / deg.T

    # Mismatch with respect to the neighborhood
    local_misalign = (sin_t - sin_neigh)**2 + (cos_t - cos_neigh)**2

    # Approximate local order parameter
    local_r = torch.sqrt(sin_neigh**2 + cos_neigh**2)

    # Normalized degree feature
    deg_feat = (deg.squeeze(-1) / deg.max()).unsqueeze(0).repeat(B, 1)

    feats = torch.stack([
        sin_t,
        cos_t,
        sin_err,
        cos_err,
        sin_neigh,
        cos_neigh,
        local_misalign,
        local_r,
    ], dim=-1)  # Stack all node-wise features into shape [B, N, 8]

    return feats



class NRPConsensusControllerTrainableNode:
    def __init__(
        self,
        neural_model,
        adjacency,
        beta: float,
        gating: bool = True,
        alpha: float = 10.0,
        k_select: int = 10,
        tau: float = 0.5,
        u_clip: float = 2.0,
        soft_selection: bool = True
    ):
        """
        Trainable node-wise NRP controller based on learned node features.

        Args:
            neural_model: Per-node neural model producing logits.
            adjacency: Graph adjacency matrix.
            beta: Consensus gain.
            gating: Whether to use the global gate.
            alpha: Gate parameter.
            k_select: Number of selected nodes in discrete mode.
            tau: Softmax temperature for soft selection.
            u_clip: Saturation bound for the control input.
            soft_selection: Whether to use differentiable soft selection.
        """
        self.model = neural_model
        self.adjacency = adjacency
        self.beta = beta
        self.gating = gating
        self.alpha = alpha
        self.k_select = k_select
        self.tau = tau
        self.u_clip = u_clip
        self.soft_selection = soft_selection

    def __call__(self, theta_measured: torch.Tensor):
        """
        Compute node-wise trainable NRP control and selection mask.

        Args:
            theta_measured: Tensor of shape [B, N] or [N].

        Returns:
            Tuple (u, m) with final control input and node mask.
        """
        squeeze_back = False
        if theta_measured.dim() == 1:
            theta = theta_measured.view(1, -1)
            squeeze_back = True
        else:
            theta = theta_measured

        # 1) Base consensus control
        info = global_order_and_mean_phase(theta)
        r = info["r"]
        psi = info["psi"]

        u_cons = -self.beta * torch.sin(theta - psi.view(-1, 1))

        if self.gating:
            gate = torch.tanh(self.alpha * (1.0 - r)).view(-1, 1)
            u_cons = u_cons * gate

        # 2) Features and logits
        node_features = build_node_features(theta, self.adjacency)   # Node-wise feature tensor [B, N, F]
        z = self.model(node_features)                                # Node logits [B, N]

        B, N = z.shape
        k = min(self.k_select, N)

        # 3) Selection mask
        if self.soft_selection:
            probs = torch.softmax(z / self.tau, dim=1)   # Soft assignment with sum equal to 1
            m = probs * k                                # Rescale so that the total mass is approximately k
            m = torch.clamp(m, 0.0, 1.0)
        else:
            m = torch.zeros_like(z)
            idx = torch.topk(z, k=k, dim=1).indices
            m.scatter_(1, idx, 1.0)

        # 4) Final masked control
        u = m * u_cons

        if self.u_clip is not None:
            u = torch.clamp(u, -self.u_clip, self.u_clip)

        return (u.squeeze(0), m.squeeze(0)) if squeeze_back else (u, m)
    

class RNMaskControllerNode(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 64):
        """
        Deep per-node mask network with tanh activations and learnable output scaling.

        Args:
            in_features: Number of input features per node.
            hidden_dim: Hidden layer width.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs):
        """
        Compute node-wise logits from batched node features.

        Args:
            inputs: Tensor of shape [B, N, F].

        Returns:
            Tensor of shape [B, N].
        """
        if inputs.dim() != 3:
            raise ValueError(f"Expected input shape [B, N, F], got {inputs.shape}")

        B, N, F = inputs.shape
        x = inputs.reshape(B * N, F)           # Flatten batch and node dimensions
        raw_output = self.net(x)               # Apply the MLP to each node independently
        raw_output = raw_output.reshape(B, N)  # Restore batch and node structure

        return raw_output * self.output_scale