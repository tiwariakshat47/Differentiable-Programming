# Differentiable Vigenere (polyalphabetic) substitution cipher
# with KNOWN LENGTH KEY
# A (partially) known plaintext attack is suitable for cryptanalysis.
# TODO: implement a ciphertext-only attack (COA)

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

# TODO: have GPT 4o summarize this section more formally
# Differentiable operations used by the Vigenere Ciphere include
# modulus, iterating/indexing/accessing the secret and the key, addition, subtraction
# but not all of these need differentiable implementation, as some are used for unrelated operations.
# High level steps:
# 1. Assume we have a plaintext input called "secret" consisting only of capital letters without whitespace.
# 2. Assume there exists some key of known/unknown/estimated length with the same alphabetic constraints.
# 3. Make a loop for every character in secret and maintain an index for the characters in the key
# 4. Calculate the shift, which is the ascii value of the 

#################################
### Discrete Helper Functions ###
#################################

# ord(char) returns unicode value of char
# Normalize the unicode values around A, so 0-25 is A-Z.

def char_to_index(c):
    """
    Given a char c, returns the 0-25 unicode value
    """
    return ord(c) - ord('A')

def index_to_char(i):
    """
    Given an integer i from 0-25, returns the character corresponding to that unicode value
    """
    if i < 0 or i > 25: print(f"ERROR: index_to_char invalid parameter, i == {i}")
    return chr(int(i) + ord('A'))

# Purpose of one hot vectors, according to GPT 4o:
# "Convert each ciphertext letter into a one-hot vector.
# Compare these one-hots against the soft distributions output by your differentiable cipher.
# Use cross-entropy loss (or log-likelihood) to measure how close the model’s prediction is to the true letter."
def one_hot(index, size=ALPHABET_SIZE):
    """
    Creates a one-hot numpy array (vector) where its i-th element is 1.0
    """
    v = np.zeros(size)
    v[index] = 1.0
    return v

# def soft_to_discrete(ciphertext_probs):
#     """
#     Convert soft probabilities at each position in the ciphertext to a discrete string using argmax
#     (argmax returns the index of the maximum value in an array)
#     """
#     # Get indices of highest probability in each row
#     discrete_indices = np.argmax(ciphertext_probs, axis=1)
#     #print(f"discrete indices: {discrete_indices}")

#     # Convert indices back to characters based on the unicode values
#     return ''.join(index_to_char(i) for i in discrete_indices)

################################
### Differentiable Functions ###
################################

# shifted_letter = (plaintext_letter + key_shift) % 26
# Above is the discrete, hard shift from the forward process of the Vigenere cipher
# Need to make this operation differentiable i.e. soft

def soft_shift(letter_index, continuous_shift, sigma=1.0):
    """
    letter_index is a position 
    Given a plaintext letter (by its index) and a continuous shift s,
    compute a soft probability distribution over the output letters.

    Therefore we return a 26-length array where the highest probability
    is assigned to the letter closest to the shifted index.
    """
    j = np.arange(ALPHABET_SIZE)  # [0, 1, 2, ..., 25]

    # target position = (letter_index + continuous_shift)
    # Compute the distance from the ideal shifted position
    # Note that the shift is not necessarily an integer
    # Use circular difference to account for going from Z back to A
    diff = np.abs(j - (letter_index + continuous_shift))
    circular_diff = np.minimum(diff, ALPHABET_SIZE - diff)

    # Apply Gaussian to obtain the weights,
    # then normalize to obtain a probability distribution
    weights = np.exp(- (circular_diff ** 2) / (2 * sigma ** 2))
    dist = weights / np.sum(weights)

    return dist

# def diff_mod(x, m):
#     """
#     Differentiable approximation of x % m using the fractional part method.
#     """
#     # Use fractional part: x - floor(x)
#     return m * (x / m - np.floor(x / m))

def vigenere_encrypt_differentiable(secret, key_shifts, sigma=1.0):
    """
    Encrypt the plaintext using continuous key shifts.
    Returns an array of shape (n, 26) where each row is a probability distribution
    over the output alphabet.
    """
    secret = secret.upper().replace(" ", "")
    plaintext_indices = np.array([char_to_index(c) for c in secret])
    output = []
    for i, p in enumerate(plaintext_indices):
        s = key_shifts[i % len(key_shifts)]
        output.append(soft_shift(p, s, sigma))
    return np.array(output)

# No differentiable decryption, not necessary for our goals since in practical applications,
# you will almost never be able to produce the correct reverse process of the encryption.


########################
### Loss Function(s) ###
########################

# The following cross-entropy loss function was made by GPT o3-mini-high
def loss_fcn(candidate_key_shifts, secret, target_onehots, sigma=0.5):
    """
    Computes the average negative log-likelihood (i.e. cross entropy loss) between the
    original ground-truth ciphertext and the generated ciphertext.
    """
    cipher_probs = vigenere_encrypt_differentiable(secret, candidate_key_shifts, sigma)
    eps = 1e-8  # numerical stability
    # For each position, select the probability corresponding to the target letter.
    probs_for_target = np.sum(cipher_probs * target_onehots, axis=1) + eps
    return -np.mean(np.log(probs_for_target))

# TODO: try more loss functions, adjust sigma depending on input size

###############
### Testing ###
###############

# TODO: update default params based on best performance
def run_test(true_secret, true_key, n_iters=200, sigma=1.0, learning_rate=0.5):
    """
    Given ground-truth values for a secret and a key, applies the discrete Vigenere cipher to obtain the true ciphertext.
    Then performs gradient descent
    """
    print(f"\n---------- Test Case ----------")
    print(f"true secret: {true_secret}")
    print(f"true key: {true_key}")
    
    # Generate the true ciphertext using discrete encryption on the true secret and true key
    true_ciphertext = vigenere_encrypt_discrete(true_secret, true_key)
    print(f"true ciphertext: {true_ciphertext}")
    
    # Convert the true ciphertext to an array of one-hot vectors
    onehot_true_ciphertext = np.array([one_hot(char_to_index(c)) for c in true_ciphertext])
    
    # TODO: try implementing with unknown/approximated key length
    # There are discrete algorithms to approximate the key length which may be worth looking into
    # but are they/can they be made differentiable?
    key_length = len(true_key)
    
    # Initialize candidate key shifts as continuous values from 0-25
    candidate_key_shifts = np.random.uniform(0, 25, key_length)
    # print(f"candidate_key_shifts: {candidate_key_shifts}")
    
    # Set up gradient computation with autograd
    loss_grad = grad(loss_fcn)
    loss_history = []
    
    # Gradient descent over n iterations, 200 by default
    for i in range(n_iters):
        current_loss = loss_fcn(candidate_key_shifts, true_secret, onehot_true_ciphertext, sigma)
        loss_history.append(current_loss)
        gradients = loss_grad(candidate_key_shifts, true_secret, onehot_true_ciphertext, sigma)
        candidate_key_shifts -= learning_rate * gradients
        
        # Print progress every _ iterations and on the final iteration
        if i % 100 == 0 or i == n_iters - 1:
            # Convert the current candidate key shifts to letters by rounding, modding to [0–25], and mapping to A–Z
            candidate_key = ''.join(index_to_char(round(s) % 26) for s in candidate_key_shifts)
            print(f"iteration {i}: loss = {current_loss:.4f}, candidate key = {candidate_key}")

    # Build the recovered key using the final shift values    
    recovered_key = ''.join(index_to_char(round(s) % 26) for s in candidate_key_shifts)
    print("recovered key:", recovered_key)

    # Perform discrete decryption using the true ciphertext and the recovered key to see what plaintext we got
    recovered_plaintext = vigenere_decrypt_discrete(true_ciphertext, recovered_key)
    print(f"recovered plaintext: {recovered_plaintext}")
    
    # Plotting the loss
    plt.figure()
    plt.plot(loss_history, label="Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Loss Evolution: True Key '{true_key}'")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    np.random.seed(42)  # fixed seed for reproducibility
    
    # Generated test cases by GPT 4o as (secret, key) pairs
    test_cases = [
        ("ATTACKATDAWN", "LEMON"),
        ("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG", "SECRET"),
        ("THISISAVERYLONGPLAINTEXTTHATWEAREUSINGTOTESTTHEGRADIENTBASEDKEYRECOVERYAPPROACH", "GRADIENT"),
        ("DIFFERENTIABLEPROGRAMMINGISFUNANDPOWERFUL", "AUTOGRAD"),
        ("THISMESSAGEISENCRYPTEDWITHAVERYLONGKEYTOTESTTHEDIFFERENTIABLEAPPROACH",
         "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    ]
    
    # BEST RESULTS SO FAR: sigma=2.0, LR=1.0
    # good sigma values to choose: 1.8, 2.0
    # Notes about results:
    # with the above hyperparameters, for the test case with key "GRADIENT" the model correctly obtains the key at ~100 iterations
    # but then obtains "HRADIENT" from ~200th iteration and onward
    # ^ resolved the above with sigma=1.8
    for secret, key in test_cases:
        run_test(secret, key, n_iters=1000, sigma=1.8, learning_rate=1.0)
