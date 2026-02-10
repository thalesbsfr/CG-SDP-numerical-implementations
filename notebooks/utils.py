from typing import Optional, Tuple, List, Dict, Callable
import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm, fractional_matrix_power, expm



def calculate_eigenvalues(matrix):

  eigenvalues = []
  threshold = 1e-15
  for i in range( np.linalg.eigvals(matrix).shape[0]):
    if abs(np.imag(np.linalg.eigvals(matrix)[i])) < threshold:
      eigenvalues.append(float(np.real(np.linalg.eigvals(matrix)[i])))

  for i in range(len(eigenvalues)):
    if abs(eigenvalues[i]) < threshold:
      eigenvalues[i] = 0

  return eigenvalues

#Defining the U_{B|A}

def unitary_channel_swap(input):
  U = np.array([[1,0,0,0],
                [0,0,1,0],
                [0,1,0,0],
                [0,0,0,1]])
  return U @ input @ U.conj().T

def sigma_z():
    """Pauli-Z matrix."""
    return np.array([[1, 0], [0, -1]])

#Letting U = e^{i \sigma_z ⊗ \sigma_z t}
def unitary_evolution(t):
    """Alternative using scipy.linalg.expm."""
    sz = sigma_z()
    sz_sz = np.kron(sz, sz)
    U = expm(1j* sz_sz * t)
    return U

#Defining the z-interaction channel
def unitary_channel_sigma_z(t=1):

  U = unitary_evolution(t)

  def unitary_channel_sigma_z_act(input):
      return U @ input @ U.conj().T

  return unitary_channel_sigma_z_act


def partial_transpose(rho, dims, subsystems_to_transpose):
    """
    Compute the partial transpose of a density matrix.

    Parameters:
        rho: numpy.ndarray
            Density matrix of the quantum state
        dims: list or tuple
            Dimensions of each subsystem (e.g., [2, 2] for two qubits)
        subsystems_to_transpose: list
            Indices of subsystems to transpose (0-based)

    Returns:
        numpy.ndarray: The partially transposed density matrix
    """
    # Reshape the density matrix into a tensor with indices for each subsystem
    tensor = rho.reshape(dims + dims)

    # Determine which indices to transpose
    n = len(dims)
    transpose_indices = list(range(2*n))
    for i in subsystems_to_transpose:
        transpose_indices[n + i], transpose_indices[i] = transpose_indices[i], transpose_indices[n + i]

    # Perform the partial transpose
    transposed_tensor = np.transpose(tensor, axes=transpose_indices)

    # Reshape back to a density matrix
    return transposed_tensor.reshape(rho.shape)

def partial_trace(input, dims: List[int], axis: Optional[int] = 0) -> np.array:
  """
  Calculate the partial trace for a 2D matrix and 2 subsystems.

  Parameters
  ----------
  input : np.array
      The 2D matrix to take the partial trace of.
  dims : tuple of ints
      A tuple of ints enconding the dimensions of each subsystem (working only with two subsystems)
  axis : int
      The index of the subsystem in which will take the partial trace.

  Returns
  -------
  np.array
      The partial trace of the input matrix.

  """
  if len(dims) > 2:
    raise ValueError("The number of subsystems exceeds 2.")
    return None

  if axis == 0:
    result = 0
    for i in range(dims[axis]):
      basis_element = np.zeros(dims[axis])
      basis_element[i] = 1
      left_kraus_op = np.kron(basis_element,np.eye(dims[axis+1]))
      result += left_kraus_op @ input @ left_kraus_op.conj().T
    return result
  else:
    result = 0
    for i in range(dims[axis]):
      basis_element = np.zeros(dims[axis])
      basis_element[i] = 1
      left_kraus_op = np.kron(np.eye(dims[axis-1]),basis_element)
      result += left_kraus_op @ input @ left_kraus_op.conj().T
    return result



def petz_recovery_map(rho_A, kraus_ops):
    """
    Implement the Petz recovery map R_A|C using Kraus operators of the channel to be recovered.

    Parameters:
    - rho_A: Initial density matrix (dim_A x dim_A)
    - kraus_ops: List of Kraus operators [K_1, K_2, ...] where each K_i is (dim_C x dim_A)

    Returns:
    - A function representing the Petz recovery map R_A|C that acts on density matrices in C
    """
    dim_A = rho_A.shape[0]
    dim_C = kraus_ops[0].shape[0] #As kraus_CG is composed by (2x4) matrices, dim_C =2, as it should!

    # Compute rho_C = E_C|A(rho_A) = sum_i K_i rho_A K_i^†
    rho_C = sum(K @ rho_A @ K.conj().T for K in kraus_ops)

    # Compute the square roots and inverse square roots
    sqrt_rho_A = fractional_matrix_power(rho_A, 0.5)
    inv_sqrt_rho_C = fractional_matrix_power(rho_C, -0.5)

    # Pre-compute the adjoint Kraus operators for the recovery map
    adjoint_kraus = [sqrt_rho_A @ K.conj().T @ inv_sqrt_rho_C for K in kraus_ops] #\rho_A^{1/2} K^{\dagger}(\rho_b^{-1/2})

    def recovery_map(input_state):
        """
        The Petz recovery map R_A|C applied to a input state

        Parameters:
        - input_state: Density matrix in system C (dim_C x dim_C)

        Returns:
        - Recovered density matrix in system A (dim_A x dim_A)
        """
        # Apply the recovery map using the adjoint Kraus operators
        #return sum(A @ sigma_C @ A.conj().T for A in adjoint_kraus)

        #Computing the action of the recovery map
        return sum(sqrt_rho_A @ K.conj().T @ inv_sqrt_rho_C @ input_state @ inv_sqrt_rho_C @ K @ sqrt_rho_A for K in kraus_ops)

    return recovery_map


#Kraus operators of the blurred and saturated detector
def kraus_CG():
  return [np.array([[1,0,0,0],
                    [0, 1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]]),
                    np.array([[0,0,0,0],
                    [0,1/np.sqrt(3),0,-1/np.sqrt(3)]]),
                    np.array([[0,0,0,0],
                    [0,1/np.sqrt(3),-1/np.sqrt(3),0]]),
                    np.array([[0,0,0,0],
                    [0,0,1/np.sqrt(3),-1/np.sqrt(3)]])]

#Defining the acting of the coarse graining map as the blurred and saturated detector
def coarse_graining_blursatur(input):
    kraus_CG_operators = kraus_CG()
    return sum(K @ input @ K.conj().T for K in kraus_CG_operators)

#===================================================================================
#Defining the acting of the coarse graining map as the partial trace over the second subsystem:

def kraus_partial_trace_B():
    ket_0 = np.array([[1],[0]])
    ket_1 = np.array([[0],[1]])

    K_0 = np.kron(np.eye(2), ket_0.T)
    K_1 = np.kron(np.eye(2),ket_1.T)
    return [K_0, K_1]

def coarse_graining_partial_trace_B(input):
  return sum(K @ input @ K.conj().T for K in kraus_partial_trace_B())


def define_all(d_A,d_B,d_C,d_D,rho_A,kraus_op, coarse_graining_map,unitary_channel):
    #Defining the Petz' recovery map for a given rho_A
    recovery_map = petz_recovery_map(rho_A, kraus_op)

    #Defining the choi states
    choi_CG = calc_choi_CG(d_A, d_C, coarse_graining_map)
    choi_U = calc_choi_U(d_A,d_B,unitary_channel)

    #Via the action of \Gamma^{petz}
    choi_Gamma_petz = calc_choi_Gamma_petz_act(d_C,d_D,recovery_map,coarse_graining_map, unitary_channel)

    return choi_CG, choi_U, choi_Gamma_petz

def generate_werner_Bell_projection (l: float) -> np.array:

  """
   Funtion to generate Werner states with the projection onto the Bell state |\psi^->

  Parameters
  ----------
  p: float
    Pameter of the Werner state between 0 and 1.

  Returns
  -------
  np.ndarray
    Werner state of dimension 4x4
  """
  return (l/2)* np.array([[0,0,0,0],[0,1,-1,0],[0,-1,1,0],[0,0,0,0]]) + ((1-l)/4)* np.eye(4)


def calc_choi_CG(d_A, d_C, coarse_graining):
    #Defining the computational basis
    ket_0 = np.array([[1],[0]])
    ket_1 = np.array([[0],[1]])

    #Computational basis of C^2 ⊗ C^2
    ket_00 = np.kron(ket_0,ket_0) #0
    ket_01 = np.kron(ket_0,ket_1) #1
    ket_10 = np.kron(ket_1,ket_0) #2
    ket_11 = np.kron(ket_1,ket_1) #3

    ket_array = [ket_00,ket_01,ket_10,ket_11]
    outer_products = []

    #Creating the 4x4 matrices from the computational basis of C^2 ⊗ C^2
    for i in range(len(ket_array)): #Setting the ket in each i-th iteration
        outer_products.append([np.kron(ket_array[i],ket_array[j].T) for j in range(len(ket_array))]) #Appending each matrix for a given i row
        #outer_products = [[|00X00|,|00X01|, |00X10|, |00X11|], [...], ...]

    #Computing the Choi state of the coarse graining map \Lambda_{C|A}
    indv_terms = []
    for i in range(len(outer_products)): #Calculating each application of the \Lambda_{C|A} individually
      indv_terms.append([np.kron(outer_products[i][j],coarse_graining(outer_products[i][j])) for j in range(len(outer_products[i]))])
      #indv_term = [[|00X00| ⊗ \Lambda_{C|A}(|00X00|),|00X01| ⊗ \Lambda_{C|A}(|00X01|), ... ], [|01X00| ⊗ \Lambda_{C|A}(|01X00|), ...], ...]

    #Adding all the terms to get the Choi state rho_{C|A} of the unitary map \Lambda_{C|A}
    choi_CG = sum(indv_terms[i][j] for i in range(len(indv_terms)) for j in range(len(indv_terms[i]))) # = |00X00| ⊗ \Lambda_{C|A}(|00X00| + |00X01| ⊗ \Lambda_{C|A}(|00X01|) + ...

    return choi_CG
    
def calc_choi_U(d_A, d_B, unitary_channel):
    #Defining the computational basis
    ket_0 = np.array([[1],[0]])
    ket_1 = np.array([[0],[1]])

    #Computational basis of C^2 ⊗ C^2
    ket_00 = cp.kron(ket_0,ket_0) #0
    ket_01 = cp.kron(ket_0,ket_1) #1
    ket_10 = cp.kron(ket_1,ket_0) #2
    ket_11 = cp.kron(ket_1,ket_1) #3

    ket_array = [ket_00.value,ket_01.value,ket_10.value,ket_11.value]
    outer_products = []

    #Creating the 4x4 matrices from the computational basis of C^2 ⊗ C^2
    for i in range(len(ket_array)): #fixing the ket in each i-th iteration
        outer_products.append([np.kron(ket_array[i],ket_array[j].T) for j in range(len(ket_array))]) #Appending each matrix for a given i row
        #outer_products = [[|00X00|,|00X01|, |00X10|, |00X11|], [...], ...]

    #Computing the Choi state of the map U_{B|A}
    indv_terms = []
    for i in range(len(outer_products)): #Calculating each application of the U_{B|A} individually
      indv_terms.append([np.kron(outer_products[i][j],unitary_channel(outer_products[i][j])) for j in range(len(outer_products[i]))])
      #indv_term = [[|00X00| ⊗ U_{B|A}(|00X00|),|00X01| ⊗ U_{B|A}(|00X01|), ... ], [|01X00| ⊗ U_{B|A}(|01X00|), ...], ...]

    #Adding all the terms to get the Choi state rho_{B|A} of the unitary map U_{B|A}
    choi_U = sum(indv_terms[i][j] for i in range(len(indv_terms)) for j in range(len(indv_terms[i]))) # = |00X00| ⊗ U_{B|A}(|00X00| + |00X01| ⊗ U_{B|A}(|00X01|) + ...

    return choi_U

def Gamma_petz(input,recovery, coarse_graining,unitary_channel):
    return coarse_graining(unitary_channel(recovery(input)))

def calc_choi_Gamma_petz_act(d_C, d_D,recovery,coarse_graining,unitary_channel):

    #Defining the computational basis of C^2
    ket_0 = np.array([[1],[0]])
    ket_1 = np.array([[0],[1]])

    ket_array = [ket_0, ket_1]
    outer_products = []

    #Creating the 2x2 matrices from the computational basis of C^2
    for i in range(len(ket_array)): #fixing the ket in each i-th iteration
        outer_products.append([np.kron(ket_array[i],ket_array[j].T) for j in range(len(ket_array))]) #Appending each matrix for a given ith row
        #outer_products = [[|0X0|,|0X1|], ... ]

    #Computing the Choi state of the \Gamma^{petz}
    indv_terms = []
    for i in range(len(outer_products)): #Calculating each application of the U_{B|A} individually
      indv_terms.append([np.kron(outer_products[i][j],Gamma_petz(outer_products[i][j],recovery,coarse_graining,unitary_channel)) for j in range(len(outer_products[i]))])
      #indv_term = [[|0X0| ⊗ \Gamma(|0X0|),|0X1| ⊗ \Gamma(|0X1|)], ... ]

    #Adding all the terms to get the Choi state rho^{petz}_{D|C} of the \Gamma^{petz}
    choi_Gamma_petz = sum(indv_terms[i][j] for i in range(len(indv_terms)) for j in range(len(indv_terms[i]))) # = |0X0| ⊗ \Gamma(|0X0|) + |0X1| ⊗ \Gamma(|0X1|) +...
    return choi_Gamma_petz