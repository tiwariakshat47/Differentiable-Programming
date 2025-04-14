# Differentiable Vigenere (polyalphabetic) substitution cipher
# with unknown key length (overparameterization) and known plaintext.
# 
# In this version, we jointly optimize:
# - candidate_key_shifts: a vector of length max_key_length (overparameterized candidate key)
# - L_eff: a learnable scalar representing the effective key length.
#
# L_eff is then used to build a differentiable mask (via a sigmoid) so that
# key positions with indices greater than L_eff are softly suppressed.
#
# We also add an additional regularization term (lambda_eff * L_eff) in the loss
# to encourage L_eff to be small, i.e. as close as possible to the true key length.

"""Imports
"""
from sub_cipher import *
import autograd.numpy as np
from autograd import grad
import string
import matplotlib.pyplot as plt

"""Globals
"""
ALPHABET = string.ascii_uppercase
ALPHABET_SIZE = 26

#################################
### Discrete Helper Functions ###
#################################

def char_to_index(c):
    """
    Given a char c, returns the 0-25 index for A-Z.
    """
    return ord(c) - ord('A')

def index_to_char(i):
    """
    Given an integer i from 0-25, returns the corresponding uppercase letter.
    """
    if i < 0 or i >= ALPHABET_SIZE:
        print(f"ERROR: index_to_char received invalid index: {i}")
    return chr(int(i) + ord('A'))

def one_hot(index, size=ALPHABET_SIZE):
    """
    Creates a one-hot vector of length `size` with 1 at position `index`.
    """
    v = np.zeros(size)
    v[index] = 1.0
    return v

################################
### Differentiable Functions ###
################################

def soft_shift(letter_index, continuous_shift, sigma=1.0):
    """
    Given a plaintext letter (by its index) and a continuous shift,
    compute a soft probability distribution over the output letters.
    
    The output is a 26-length vector where the highest probability is given to
    the letter closest to (letter_index + continuous_shift) modulo 26.
    sigma controls the sharpness: larger sigma results in a smoother distribution.
    """
    j = np.arange(ALPHABET_SIZE)  # [0,1,...,25]
    diff = np.abs(j - (letter_index + continuous_shift))
    circular_diff = np.minimum(diff, ALPHABET_SIZE - diff)
    weights = np.exp(- (circular_diff ** 2) / (2 * sigma ** 2))
    dist = weights / np.sum(weights)
    return dist

def get_key_mask(max_key_length, L_eff, a=10.0):
    """
    For candidate key indices 0,1,...,max_key_length-1, compute a differentiable mask.
    The mask is defined by 1/(1+exp(a*(j - L_eff))) which is near 1 when j << L_eff and near 0 when j >> L_eff.
    a controls the sharpness of this transition.
    """
    j = np.arange(max_key_length)
    mask = 1.0 / (1.0 + np.exp(a * (j - L_eff)))
    return mask

def vigenere_encrypt_differentiable(secret, candidate_key_shifts, L_eff, sigma=1.0):
    """
    Encrypt the plaintext using candidate key shifts and the learned effective key length L_eff.
    candidate_key_shifts is a vector of length max_key_length.
    L_eff is a differentiable scalar (effective key length).
    
    First, candidate_key_shifts is weighted by a differentiable mask computed from L_eff.
    Then, for each plaintext letter, we select the effective key shift using cyclic (integer) indexing.
    
    Returns an array of shape (n, 26), where each row is a soft probability distribution over the output alphabet.
    """
    secret = secret.upper().replace(" ", "")
    plaintext_indices = np.array([char_to_index(c) for c in secret])
    max_key_length = candidate_key_shifts.shape[0]
    mask = get_key_mask(max_key_length, L_eff, a=10.0)
    effective_key_shifts = candidate_key_shifts * mask
    output = []
    for i, p in enumerate(plaintext_indices):
        j = i % max_key_length  # cyclic indexing (non-differentiable, but acceptable)
        s = effective_key_shifts[j]
        output.append(soft_shift(p, s, sigma))
    return np.array(output)

################################
### Loss Function(s) ###
################################

def loss_fcn(candidate_key_shifts, L_eff, true_secret, target_onehots, sigma=0.5, lambda_eff=0.2):
    """
    Computes the loss as the cross-entropy between the generated ciphertext (from encrypting
    true_secret with the candidate key) and the target ciphertext (target_onehots),
    plus a regularization term that penalizes large effective key length.
    
    The regularization term is defined as lambda_eff * L_eff, which encourages L_eff to be small.
    """
    cipher_probs = vigenere_encrypt_differentiable(true_secret, candidate_key_shifts, L_eff, sigma)
    eps = 1e-8
    probs_for_target = np.sum(cipher_probs * target_onehots, axis=1) + eps
    ce_loss = -np.mean(np.log(probs_for_target))
    reg_loss = lambda_eff * L_eff  # Directly penalize larger L_eff
    return ce_loss + reg_loss

################################
### Testing / Optimization ###
################################

def run_test(true_secret, true_key, n_iters=1000, sigma=2.0, learning_rate=0.5, max_key_length=50):
    """
    Runs a test with unknown key length. Even though the true key length is known for evaluation,
    we overparameterize candidate_key_shifts to length max_key_length and learn L_eff.
    
    The optimizer jointly updates candidate_key_shifts (vector of continuous shifts) and L_eff
    (effective key length) to minimize the loss between the generated ciphertext (from encrypting
    true_secret) and the true ciphertext.
    """
    print(f"\n---------- Test Case ----------")
    print(f"true secret: {true_secret}")
    print(f"true key: {true_key}")
    
    # Generate the true ciphertext using discrete Vigenère encryption.
    true_ciphertext = vigenere_encrypt_discrete(true_secret, true_key)
    print(f"true ciphertext: {true_ciphertext}")
    
    onehot_true_ciphertext = np.array([one_hot(char_to_index(c)) for c in true_ciphertext])
    
    # Initialize candidate_key_shifts overparameterized to max_key_length.
    candidate_key_shifts = np.random.uniform(0, 25, max_key_length)
    # Initialize L_eff to a value between 1 and max_key_length; we start at half.
    L_eff = np.array(max_key_length / 2.0)
    
    # Set up gradient computation with autograd.
    grad_key = grad(loss_fcn, 0)    # gradient with respect to candidate_key_shifts
    grad_L_eff = grad(loss_fcn, 1)    # gradient with respect to L_eff
    
    loss_history = []
    
    for i in range(n_iters):
        current_loss = loss_fcn(candidate_key_shifts, L_eff, true_secret, onehot_true_ciphertext, sigma, lambda_eff=0.2)
        loss_history.append(current_loss)
        g_key = grad_key(candidate_key_shifts, L_eff, true_secret, onehot_true_ciphertext, sigma, lambda_eff=0.2)
        g_L_eff = grad_L_eff(candidate_key_shifts, L_eff, true_secret, onehot_true_ciphertext, sigma, lambda_eff=0.2)
        
        candidate_key_shifts -= learning_rate * g_key
        L_eff -= learning_rate * g_L_eff
        
        # Clamp L_eff to be within [1, max_key_length]
        L_eff = np.maximum(1.0, np.minimum(L_eff, max_key_length))
        
        if i % 100 == 0 or i == n_iters - 1:
            mask = get_key_mask(max_key_length, L_eff, a=10.0)
            effective_key_shifts = candidate_key_shifts * mask
            current_eff_length = int(round(L_eff))
            # For display, use only the first current_eff_length positions.
            candidate_key_display = ''.join(index_to_char(round(s) % 26) for s in effective_key_shifts[:current_eff_length])
            print(f"iteration {i}: loss = {current_loss:.4f}, candidate key ≈ {candidate_key_display}, L_eff = {L_eff:.2f}")
    
    mask = get_key_mask(max_key_length, L_eff, a=10.0)
    effective_key_shifts = candidate_key_shifts * mask
    current_eff_length = int(round(L_eff))
    recovered_key = ''.join(index_to_char(round(s) % 26) for s in effective_key_shifts[:current_eff_length])
    print("recovered key:", recovered_key)
    
    recovered_plaintext = vigenere_decrypt_discrete(true_ciphertext, recovered_key)
    print(f"recovered plaintext: {recovered_plaintext}")
    
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, label="Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Loss Evolution: True Key '{true_key}' (Unknown key length)")
    plt.legend()
    plt.show()

################################
### Main Execution  ###
################################

if __name__ == '__main__':
    np.random.seed(42)  # fixed seed for reproducibility
    
    test_cases = [
        ("ATTACKATDAWN", "LEMON"),
        ("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG", "SECRET"),
        ("THISISAVERYLONGPLAINTEXTTHATWEAREUSINGTOTESTTHEGRADIENTBASEDKEYRECOVERYAPPROACH", "GRADIENT"),
        ("DIFFERENTIABLEPROGRAMMINGISFUNANDPOWERFUL", "AUTOGRAD"),
        ("THISMESSAGEISENCRYPTEDWITHAVERYLONGKEYTOTESTTHEDIFFERENTIABLEAPPROACH",
         "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    ]
    
    for secret, key in test_cases:
        run_test(secret, key, n_iters=1000, sigma=2.0, learning_rate=0.5, max_key_length=50)
