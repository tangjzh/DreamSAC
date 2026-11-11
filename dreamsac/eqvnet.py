import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

try:
    import e3nn_jax as e3nn
    from e3nn_jax import Irreps, spherical_harmonics
    E3NN_AVAILABLE = True
except ImportError:
    E3NN_AVAILABLE = False
    print("Warning: e3nn-jax not available. Install with: pip install e3nn-jax")

import embodied.jax.nets as nn

f32 = jnp.float32


class EquivariantHamiltonianNet(nj.Module):
    """
    Rotationally invariant Hamiltonian network using e3nn-jax.
    
    The network maps (q, p, action) → scalar energy H, where:
    - H(R·q, R·p) = H(q, p) for any rotation R ∈ SO(3)
    
    This is achieved by using:
    1. Spherical harmonics for feature extraction
    2. Equivariant linear layers
    3. Tensor products for combining features
    4. Invariant aggregation for final scalar output
    """
    
    layers: int = 3
    hidden_channels: int = 64
    max_ell: int = 2  # Maximum spherical harmonic degree (l=0,1,2)
    norm: str = 'rms'
    act: str = 'silu'
    use_e3nn: bool = True
    
    def __init__(self, **kw):
        # Filter out equivariant-specific parameters that shouldn't go to base layers
        filtered_kw = {k: v for k, v in kw.items() if k not in ['max_ell', 'use_e3nn', 'use_equivariant']}
        self.kw = filtered_kw
        if self.use_e3nn and not E3NN_AVAILABLE:
            raise ImportError("e3nn-jax is missing. Use the dreamsac_full profile or install e3nn-jax.")
        if self.use_e3nn and not hasattr(e3nn, 'Linear'):
            raise NotImplementedError(
                "dreamsac_e3nn requires an e3nn.Linear API unavailable in this installation. "
                "Use the dreamsac_full profile.")
        if self.use_e3nn and E3NN_AVAILABLE:
            # Define the Irreducible Representations (Irreps) for the network
            
            # The hidden layers will contain a mix of scalars (0e) and vectors (1o)
            # This is a common choice for building expressive equivariant networks.
            irreps_hidden = e3nn.Irreps(f"{self.hidden_channels}x0e + {self.hidden_channels//2}x1o")

            # The final output must be a single scalar: l=0, even parity -> "1x0e"
            irreps_out = e3nn.Irreps("1x0e")
            
            # Define the equivariant linear layers
            # This is a simplified equivariant MLP. A more advanced network (e.g., for graph tasks)
            # would use e3nn.TensorProduct to combine features.
            
            # The input to the first layer will be the aggregated vectors ("1o")
            self.lin1 = e3nn.Linear(e3nn.Irreps("1o"), irreps_hidden)
            self.lin2 = e3nn.Linear(irreps_hidden, irreps_hidden)
            self.lin_out = e3nn.Linear(irreps_hidden, irreps_out)
    
    def __call__(self, q, p, action):
        """
        Compute rotationally invariant Hamiltonian.
        
        Args:
            q: Generalized coordinates [B, q_dim, classes]
            p: Generalized momenta [B, p_dim, classes]
            action: Action vector [B, action_dim] or dict
        
        Returns:
            energy: Scalar Hamiltonian [B]
        """
        if not self.use_e3nn:
            return self._mlp_hamiltonian(q, p, action)
        else:
            return self._equivariant_hamiltonian(q, p, action)
    
    def _mlp_hamiltonian(self, q, p, action):
        """Fallback: Standard MLP (not equivariant)."""
        # Ensure correct dtype
        q = nn.cast(q)
        p = nn.cast(p)
        
        # Flatten
        q_flat = q.reshape(q.shape[0], -1)
        p_flat = p.reshape(p.shape[0], -1)
        
        # Handle action - concatenate if dict
        if isinstance(action, dict):
            action = jnp.concatenate([v.reshape(v.shape[0], -1) for v in action.values()], axis=-1)
        
        action = nn.cast(action)
        
        # Concatenate
        x = jnp.concatenate([q_flat, p_flat, action], axis=-1)
        
        # MLP
        for i in range(self.layers):
            x = self.sub(f'mlp{i}', nn.Linear, self.hidden_channels, **self.kw)(x)
            x = nn.act(self.act)(x)
            if i < self.layers - 1:
                x = self.sub(f'mlp{i}norm', nn.Norm, self.norm)(x)
        
        # Scalar output
        energy = self.sub('mlp_out', nn.Linear, 1, **self.kw)(x)
        return energy.squeeze(-1)
    
    def _equivariant_hamiltonian(self, q, p, action):
        """
        Equivariant network using e3nn-jax.
        
        Strategy:
        1. Treat q and p as 3D positions/vectors
        2. Extract spherical harmonic features (rotationally equivariant)
        3. Process through equivariant layers
        4. Aggregate to scalar (invariant output)
        """
        batch_size = q.shape[0]
        
        # --- Step 1: Construct input vectors (same as your original code) ---
        # This reshapes the flat latent q and p into a set of artificial 3D vectors.
        q_flat = q.reshape(batch_size, -1)
        p_flat = p.reshape(batch_size, -1)
        qp_flat = jnp.concatenate([q_flat, p_flat], axis=-1)
        
        total_dim = qp_flat.shape[-1]
        num_points = (total_dim + 2) // 3
        padding = num_points * 3 - total_dim
        if padding > 0:
            qp_flat = jnp.pad(qp_flat, [[0, 0], [0, padding]])
        
        positions = qp_flat.reshape(batch_size, num_points, 3) # Shape: [B, N, 3]

        # --- Step 2: Convert the input into an e3nn.IrrepsArray ---
        # We treat the input as vectors, which have the irrep "1o" (l=1, odd parity).
        # The IrrepsArray object tracks both the data and its transformation properties.
        x = e3nn.IrrepsArray("1o", positions) # x.shape is (B, N, 3)

        # --- Step 3: Process through equivariant layers ---
        # This replaces the manual feature engineering and standard MLP.
        
        # To create a single representation for the system, we aggregate across the "particle" dimension.
        # e3nn.mean is an equivariant operation that preserves the irreps structure.
        # x: [B, N, 3] with irreps "1o"  -->  agg_x: [B, 3] with irreps "1o"
        agg_x = e3nn.mean(x, axis=1)

        # Now, pass the aggregated features through a small equivariant MLP.
        # Layer 1: maps input vectors ("1o") to hidden representations (irreps_hidden)
        h = self.lin1(agg_x)
        # Apply a gated activation function. This is a standard equivariant nonlinearity in e3nn.
        # It uses the scalar parts of `h` to gate the non-scalar parts, preserving equivariance.
        h = e3nn.gate(h)

        # Layer 2: maps hidden representations to new hidden representations
        h = self.lin2(h)
        h = e3nn.gate(h)
        
        # --- Step 4: Output an invariant scalar ---
        # The final layer projects all hidden features onto a single scalar ("1x0e").
        # The output `energy` is guaranteed to be rotationally invariant.
        energy = self.lin_out(h) # energy.shape is (B, 1), with irreps "1x0e"
        
        # (Optional) Fuse the action features.
        # Actions are typically assumed to be invariant. They can be combined with the
        # scalar energy output before the final prediction.
        if action is not None:
             if isinstance(action, dict):
                 action_flat = jnp.concatenate([v.reshape(v.shape[0], -1) for v in action.values()], axis=-1)
             else:
                 action_flat = action.reshape(batch_size, -1)
            
             # Concatenate the invariant energy scalar with the invariant action features.
             combined_features = jnp.concatenate([energy.array, action_flat], axis=-1)
             # A final standard MLP can process these invariant features.
             final_energy = self.sub('final_mlp', nn.Linear, 1, **self.kw)(combined_features)
             return final_energy.squeeze(-1)

        # If there is no action, return the scalar energy directly.
        # .array extracts the raw jax.numpy array from the IrrepsArray object.
        return energy.array.squeeze(-1)
    
    def _compute_spherical_harmonics(self, directions):
        """
        Compute spherical harmonics for directions.
        
        Args:
            directions: [B, N, N, 3] unit vectors
        
        Returns:
            sh_features: [B, N, N, num_sh] spherical harmonic features
        """
        # For simplicity, use low-order polynomials of (x,y,z)
        # These form a rotation-equivariant basis
        
        # l=0: constant (scalar)
        l0 = jnp.ones((*directions.shape[:-1], 1))  # [B, N, N, 1]
        
        # l=1: linear (vector)
        l1 = directions  # [B, N, N, 3]
        
        # l=2: quadratic (second-order)
        x, y, z = directions[..., 0:1], directions[..., 1:2], directions[..., 2:3]
        l2 = jnp.concatenate([
            x*y, y*z, z*x,  # Off-diagonal
            x*x - y*y, 2*z*z - x*x - y*y  # Diagonal traceless
        ], axis=-1)  # [B, N, N, 5]
        
        if self.max_ell == 0:
            return l0
        elif self.max_ell == 1:
            return jnp.concatenate([l0, l1], axis=-1)
        else:  # max_ell >= 2
            return jnp.concatenate([l0, l1, l2], axis=-1)
    
    def _rbf_encoding(self, distances, num_basis=16, cutoff=5.0):
        """
        Radial basis function encoding of distances.
        
        Args:
            distances: [B, N, N] pairwise distances
            num_basis: Number of RBF basis functions
            cutoff: Cutoff distance
        
        Returns:
            rbf: [B, N, N, num_basis] RBF features
        """
        # Gaussian RBF
        centers = jnp.linspace(0, cutoff, num_basis)
        width = cutoff / num_basis
        
        # Compute RBF: exp(-((d - center) / width)^2)
        distances_expanded = distances[..., None]  # [B, N, N, 1]
        rbf = jnp.exp(-((distances_expanded - centers) / width) ** 2)
        
        return rbf


class SimplifiedEquivariantHamiltonian(nj.Module):
    """
    Simplified equivariant Hamiltonian using rotation-invariant features.
    
    This is a lightweight alternative that doesn't require e3nn-jax,
    but still respects rotational symmetry by using invariant features:
    - Norms: |q|, |p|
    - Dot products: q·p
    - Angle-independent combinations
    """
    
    layers: int = 3
    hidden_channels: int = 512
    norm: str = 'rms'
    act: str = 'silu'
    
    def __init__(self, **kw):
        # Filter out equivariant-specific parameters that shouldn't go to base layers
        filtered_kw = {k: v for k, v in kw.items() if k not in ['max_ell', 'use_e3nn', 'use_equivariant']}
        self.kw = filtered_kw
    
    def __call__(self, q, p, action):
        """
        Compute Hamiltonian using rotation-invariant features.
        
        Args:
            q: [B, q_dim] or [B, q_dim, classes] (continuous values)
            p: [B, p_dim] or [B, p_dim, classes] (continuous values)
            action: [B, action_dim] or dict
        
        Returns:
            energy: [B]
        """
        batch_size = q.shape[0]
        
        # Ensure correct dtype from the start
        q = nn.cast(q)
        p = nn.cast(p)
        
        # Flatten
        q_flat = q.reshape(batch_size, -1)  # [B, D_q]
        p_flat = p.reshape(batch_size, -1)  # [B, D_p]
        
        # Rotation-invariant features
        features = self._extract_invariant_features(q_flat, p_flat)
        
        # Ensure features are in correct dtype
        features = nn.cast(features)
        
        # Add action (assumed to be orientation-independent or in body frame)
        if isinstance(action, dict):
            action = jnp.concatenate([v.reshape(v.shape[0], -1) for v in action.values()], axis=-1)
        
        action = nn.cast(action)
        x = jnp.concatenate([features, action], axis=-1)
        
        # MLP on invariant features
        for i in range(self.layers):
            x = self.sub(f'inv{i}', nn.Linear, self.hidden_channels, **self.kw)(x)
            x = nn.act(self.act)(x)
            if i < self.layers - 1:
                x = self.sub(f'inv{i}norm', nn.Norm, self.norm)(x)
        
        # Scalar output
        energy = self.sub('inv_out', nn.Linear, 1, **self.kw)(x)
        return energy.squeeze(-1)
    
    def _extract_invariant_features(self, q, p):
        """
        Extract rotation-invariant features from q and p.
        
        Strategy:
        - Use norms (scalars)
        - Use inner products
        - Use pairwise distances in embedding space
        
        These are all invariant under SO(3) rotations.
        """
        features = []
        
        # 1. Norms (kinetic and potential energy proxies)
        q_norm = jnp.sqrt(jnp.sum(q ** 2, axis=-1, keepdims=True) + 1e-6)  # [B, 1]
        p_norm = jnp.sqrt(jnp.sum(p ** 2, axis=-1, keepdims=True) + 1e-6)  # [B, 1]
        features.extend([q_norm, p_norm])
        
        # 2. Squared norms (quadratic terms)
        q_norm_sq = (q ** 2).sum(axis=-1, keepdims=True)
        p_norm_sq = (p ** 2).sum(axis=-1, keepdims=True)
        features.extend([q_norm_sq, p_norm_sq])
        
        # 3. Dot product (interaction term)
        if q.shape[-1] == p.shape[-1]:
            qp_dot = (q * p).sum(axis=-1, keepdims=True)
            features.append(qp_dot)
        
        # 4. Cross terms (higher-order invariants)
        # |q|^2 * |p|^2
        cross_term = q_norm_sq * p_norm_sq
        features.append(cross_term)
        
        # 5. Statistical moments (permutation invariant)
        # q_mean = q.mean(axis=-1, keepdims=True)
        # p_mean = p.mean(axis=-1, keepdims=True)
        # q_std = q.std(axis=-1, keepdims=True)
        # p_std = p.std(axis=-1, keepdims=True)
        # features.extend([q_mean, p_mean, q_std, p_std])
        
        # Concatenate all invariant features
        return jnp.concatenate(features, axis=-1)


def create_hamiltonian_network(
    use_equivariant=True,
    use_e3nn=False,
    **kwargs
):
    """
    Factory function to create appropriate Hamiltonian network.
    
    Args:
        use_equivariant: If True, use equivariant/invariant features
        use_e3nn: If True and available, use e3nn-jax (slower but most accurate)
        **kwargs: Network hyperparameters
    
    Returns:
        Hamiltonian network module
    """
    if not use_equivariant:
        # Standard MLP (no symmetry)
        return EquivariantHamiltonianNet(use_e3nn=False, **kwargs)
    elif use_e3nn and E3NN_AVAILABLE:
        # Full e3nn-jax equivariant network
        return EquivariantHamiltonianNet(use_e3nn=True, **kwargs)
    else:
        # Simplified invariant features (fast, no dependencies)
        return SimplifiedEquivariantHamiltonian(**kwargs)

