import jax
import jax.numpy as jnp
import ninjax as nj

f32 = jnp.float32


class HamiltonianDynamics:
  """
  Computes time derivatives from Hamiltonian via symplectic gradient.
  
  In Hamiltonian mechanics:
    dq/dt = ∂H/∂p
    dp/dt = -∂H/∂q
  
  Or in compact form: dz/dt = S·∇H where z=(q,p) and S is the symplectic matrix.
  """
  
  def __init__(self, hamiltonian_fn):
    """
    Args:
      hamiltonian_fn: Function that computes H(q, p, action) → scalar
    """
    self.hamiltonian_fn = hamiltonian_fn
    self.nfe = 0  # Number of function evaluations (for debugging)
  
  def __call__(self, qp_state, action):
    """
    Compute time derivatives dq/dt and dp/dt.
    
    Args:
      qp_state: dict with 'q' and 'p' arrays
      action: action array
    
    Returns:
      dict with 'dq_dt' and 'dp_dt'
    """
    self.nfe += 1
    
    # Extract q and p
    q = qp_state['q']
    p = qp_state['p']
    
    # Flatten for gradient computation
    q_flat = q.reshape(q.shape[0], -1)  # [B, q_dim * classes]
    p_flat = p.reshape(p.shape[0], -1)  # [B, p_dim * classes]
    
    # Compute Hamiltonian gradients
    def H_flat(q_flat, p_flat):
      # Reconstruct structured form
      q_struct = q_flat.reshape(q.shape)
      p_struct = p_flat.reshape(p.shape)
      feat = {'q': q_struct, 'p': p_struct}
      return self.hamiltonian_fn(feat, action).sum()
    
    # Compute gradients: ∂H/∂q and ∂H/∂p
    grad_q, grad_p = jax.grad(H_flat, argnums=(0, 1))(q_flat, p_flat)
    
    # Hamilton's equations:
    # dq/dt = ∂H/∂p
    # dp/dt = -∂H/∂q
    dq_dt = grad_p.reshape(q.shape)
    dp_dt = -grad_q.reshape(p.shape)
    
    return {'dq_dt': dq_dt, 'dp_dt': dp_dt}


def leapfrog_step(hamiltonian_fn, q, p, action, dt=0.1, input_matrix_fn=None):
  """
  Leapfrog (Störmer-Verlet) symplectic integrator with external forcing.
  This is a second-order method that preserves symplectic structure.
  
  Extended to support controlled Hamiltonian dynamics (Eq. 1):
    dq/dt = ∂H/∂p
    dp/dt = -∂H/∂q + g(q)·a_t
  
  Algorithm:
    1. p_{n+1/2} = p_n - (dt/2) * ∂H/∂q(q_n) + (dt/2) * g(q_n)·a_t
    2. q_{n+1} = q_n + dt * ∂H/∂p(p_{n+1/2})
    3. p_{n+1} = p_{n+1/2} - (dt/2) * ∂H/∂q(q_{n+1}) + (dt/2) * g(q_{n+1})·a_t
  
  Args:
    hamiltonian_fn: Function H(feat, action) → energy
    q: Current generalized coordinates [B, q_dim] (continuous values)
    p: Current generalized momenta [B, p_dim] (continuous values)
    action: Action array
    dt: Time step size
    input_matrix_fn: Optional function g(q, action) → generalized force [B, p_dim]
  
  Returns:
    (q_next, p_next): Updated state
  """
  
  # q and p are continuous values [B, dim]
  # Get shapes (these are static in the trace)
  q_shape = q.shape
  p_shape = p.shape
  
  # Flatten for gradient computation
  q_flat = q.reshape(q.shape[0], -1)
  p_flat = p.reshape(p.shape[0], -1)
  
  # Define gradient function that takes action as parameter to avoid closure issues
  def H_for_grad(q_flat, p_flat, action_arg):
    # Reshape using original shapes
    q_cont = q_flat.reshape(q_shape)
    p_cont = p_flat.reshape(p_shape)
    feat = {'q': q_cont, 'p': p_cont}
    return hamiltonian_fn(feat, action_arg).sum()
  
  # Step 1: Half-step momentum update with external forcing
  grad_q = jax.grad(H_for_grad, argnums=0)(q_flat, p_flat, action)
  force = jnp.zeros_like(p_flat)
  if input_matrix_fn is not None:
    q_cont = q_flat.reshape(q_shape)
    force = input_matrix_fn(q_cont, action).reshape(p_flat.shape[0], -1)
  p_half = p_flat - 0.5 * dt * grad_q + 0.5 * dt * force
  
  # Step 2: Full-step position update
  grad_p_half = jax.grad(H_for_grad, argnums=1)(q_flat, p_half, action)
  q_next_flat = q_flat + dt * grad_p_half
  
  # Step 3: Half-step momentum update with external forcing
  grad_q_next = jax.grad(H_for_grad, argnums=0)(q_next_flat, p_half, action)
  force_next = jnp.zeros_like(p_flat)
  if input_matrix_fn is not None:
    q_next_cont = q_next_flat.reshape(q_shape)
    force_next = input_matrix_fn(q_next_cont, action).reshape(p_flat.shape[0], -1)
  p_next_flat = p_half - 0.5 * dt * grad_q_next + 0.5 * dt * force_next
  
  # Reshape back to original shapes
  q_next = q_next_flat.reshape(q_shape)
  p_next = p_next_flat.reshape(p_shape)
  
  return q_next, p_next


def euler_symplectic_step(hamiltonian_fn, q, p, action, dt=0.1, input_matrix_fn=None):
  """
  Symplectic Euler method (first-order) with external forcing.
  Simpler but less accurate than leapfrog.
  
  Extended to support controlled Hamiltonian dynamics:
    dq/dt = ∂H/∂p
    dp/dt = -∂H/∂q + g(q)·a_t
  
  Algorithm:
    1. p_{n+1} = p_n - dt * ∂H/∂q(q_n) + dt * g(q_n)·a_t
    2. q_{n+1} = q_n + dt * ∂H/∂p(p_{n+1})
  
  Args:
    Same as leapfrog_step
  
  Returns:
    (q_next, p_next): Updated state
  """
  
  # Get shapes (static in trace)
  q_shape = q.shape
  p_shape = p.shape
  
  q_flat = q.reshape(q.shape[0], -1)
  p_flat = p.reshape(p.shape[0], -1)
  
  def H_for_grad(q_flat, p_flat, action_arg):
    q_struct = q_flat.reshape(q_shape)
    p_struct = p_flat.reshape(p_shape)
    feat = {'q': q_struct, 'p': p_struct}
    return hamiltonian_fn(feat, action_arg).sum()
  
  # Update momentum with external forcing
  grad_q = jax.grad(H_for_grad, argnums=0)(q_flat, p_flat, action)
  force = jnp.zeros_like(p_flat)
  if input_matrix_fn is not None:
    force = input_matrix_fn(q, action).reshape(p_flat.shape)
  p_next_flat = p_flat - dt * grad_q + dt * force
  
  # Update position
  grad_p_next = jax.grad(H_for_grad, argnums=1)(q_flat, p_next_flat, action)
  q_next_flat = q_flat + dt * grad_p_next
  
  # Reshape
  q_next = q_next_flat.reshape(q_shape)
  p_next = p_next_flat.reshape(p_shape)
  
  return q_next, p_next


def rk4_symplectic_step(hamiltonian_fn, q, p, action, dt=0.1):
  """
  RK4 applied to Hamiltonian dynamics (not strictly symplectic but accurate).
  This is useful when you need higher accuracy and don't strictly require
  symplectic structure preservation.
  
  Args:
    Same as leapfrog_step
  
  Returns:
    (q_next, p_next): Updated state
  """
  dynamics = HamiltonianDynamics(hamiltonian_fn)
  
  def compute_derivatives(q, p):
    qp_state = {'q': q, 'p': p}
    derivs = dynamics(qp_state, action)
    return derivs['dq_dt'], derivs['dp_dt']
  
  # k1
  dq1, dp1 = compute_derivatives(q, p)
  
  # k2
  q_k2 = q + 0.5 * dt * dq1
  p_k2 = p + 0.5 * dt * dp1
  dq2, dp2 = compute_derivatives(q_k2, p_k2)
  
  # k3
  q_k3 = q + 0.5 * dt * dq2
  p_k3 = p + 0.5 * dt * dp2
  dq3, dp3 = compute_derivatives(q_k3, p_k3)
  
  # k4
  q_k4 = q + dt * dq3
  p_k4 = p + dt * dp3
  dq4, dp4 = compute_derivatives(q_k4, p_k4)
  
  # Combine
  q_next = q + (dt / 6.0) * (dq1 + 2*dq2 + 2*dq3 + dq4)
  p_next = p + (dt / 6.0) * (dp1 + 2*dp2 + 2*dp3 + dp4)
  
  return q_next, p_next


def integrate_trajectory(hamiltonian_fn, q0, p0, actions, method='leapfrog', dt=0.1):
    """
    Integrate Hamiltonian dynamics over a trajectory using jax.lax.scan for performance.
    """
    integrators = {
        'leapfrog': leapfrog_step,
        'euler': euler_symplectic_step,
        'rk4': rk4_symplectic_step,
    }
    integrator = integrators[method]

    # The function to be applied at each step of the scan.
    def step_fn(carry, action_t):
        q_current, p_current = carry
        q_next, p_next = integrator(hamiltonian_fn, q_current, p_current, action_t, dt)
        # The new carry is the next state, the output is also the next state.
        return (q_next, p_next), (q_next, p_next)

    # Initial state for the scan.
    initial_carry = (q0, p0)
    
    # Run the scan over the actions sequence.
    # The first return value is the final state, the second is the trajectory of states.
    _, (q_traj, p_traj) = jax.lax.scan(step_fn, initial_carry, actions.transpose(1, 0, 2)) # Transpose actions to [T, B, ...]
    
    # Transpose trajectories back to [B, T, ...] and prepend the initial state.
    q_traj = jnp.concatenate([q0[:, None], q_traj.transpose(1, 0, 2)], axis=1)
    p_traj = jnp.concatenate([p0[:, None], p_traj.transpose(1, 0, 2)], axis=1)
    
    return q_traj, p_traj


def compute_energy_conservation_error(hamiltonian_fn, q_traj, p_traj, actions):
    """
    Compute energy conservation error using jax.vmap for performance.
    """
    # Transpose data so the time dimension is first, for vmap to iterate over.
    q_traj_T_first = jnp.swapaxes(q_traj, 0, 1)
    p_traj_T_first = jnp.swapaxes(p_traj, 0, 1)
    actions_T_first = jnp.swapaxes(actions, 0, 1)

    # vmap the energy calculation over the time dimension.
    # The function now takes slices of the trajectory and computes energy for all time steps in parallel.
    def compute_energy_at_t(q_t, p_t, action_t):
        feat_t = {'q': q_t, 'p': p_t}
        return hamiltonian_fn(feat_t, action_t)
        
    energies_T_first = jax.vmap(compute_energy_at_t)(q_traj_T_first, p_traj_T_first, actions_T_first)
    energies = jnp.swapaxes(energies_T_first, 0, 1) # Transpose back to [B, T]

    # The rest of the logic is the same.
    energy_drift = jnp.abs(energies - energies[:, 0:1]).mean()
    energy_std = energies.std(axis=1).mean()
    
    return energy_drift, energy_std

