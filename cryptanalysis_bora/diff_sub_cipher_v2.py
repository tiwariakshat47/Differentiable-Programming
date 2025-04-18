# Differentiable Vigenere (polyalphabetic) substitution cipher
# with KNOWN LENGTH KEY
# A (partially) known plaintext attack is suitable for cryptanalysis.
# TODO: implement a ciphertext-only attack (COA)
# TODO: implement a unknown key length attack

"""Imports
"""
from sub_cipher import *
import autograd.numpy as np
import numpy #as np
from autograd import grad
import string
import matplotlib.pyplot as plt

"""Globals
"""
ALPHABET = string.ascii_uppercase
ALPHABET_SIZE = 26
MAX_KEY_LENGTH = 20  # Assumed max length of the key for estimation with unknown key length

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


def estimate_key_length(plaintext, ciphertext, max_key_length=MAX_KEY_LENGTH):
    """
    This function estimates the key length assuming both the plaintext and ciphertext are fully known.
    In the context of a known-plaintext attack, we can use the known plaintext to help us estimate the key length.
    It will almost always be exactly correct except in some cases:

    GPT 4o's analysis of edge cases:
        - Key is a repeated substring (e.g., "ABCABC" is actually "ABC" repeated): Then both 3 and 6 will produce low-variance groupings.
        - If your plaintext is shorter than, say, 2× the key length, it might not give enough data to distinguish patterns confidently.

    Try key lengths from 1 to max_len.  For each L:
      • Group shifts = (c_i - p_i) mod 26 by i % L
      • Compute stddev of each group
      • Score = mean(stddevs)
    Returns:
      best_L: the L with lowest score
      scores: list of (L, score) for all trials
    """
    # sanitize inputs
    plaintext  = plaintext.upper().replace(" ", "")
    ciphertext = ciphertext.upper().replace(" ", "")

    N = min(len(plaintext), len(ciphertext))
    scores = []

    for L in range(1, max_key_length + 1):
        # accumulate shifts per column
        cols = [[] for _ in range(L)]
        for i in range(N):
            p_idx = char_to_index(plaintext[i])
            c_idx = char_to_index(ciphertext[i])
            shift = (c_idx - p_idx) % ALPHABET_SIZE
            cols[i % L].append(shift)

        # compute per‑column stddev, ignoring empty
        stds = [numpy.std(col) for col in cols if len(col) > 1]
        score = numpy.mean(stds) if stds else numpy.inf
        scores.append((L, float(score)))

    # pick minimal
    best_L = min(scores, key=lambda x: x[1])[0]

    # return tuple of the best L based on how low the stddev was
    # and the list of all scores for each L
    return best_L, scores



#  TODO: implement this function
def kasiski_examination(ciphertext):
    """
    This function implements Kasiski examination to estimate the key length of a Vigenere cipher.
    It looks for repeated sequences of characters in the ciphertext and calculates the distances between them.
    Only ciphertext is required as input.
    """
    return

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
    # Create a reference of letter indices to compare with the soft-shifted result
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
def loss_fn(candidate_key_shifts, secret, target_onehots, sigma=0.5):
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
# Us
def run_test(true_secret, true_key, n_iters=200, sigma=1.0, learning_rate=0.5, max_key_length=MAX_KEY_LENGTH):
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
    # key_length = len(true_key)
    # Instead of "cheating" and using the true key length, we can use a random length and optimize this
    # using gradients as part of the SGD loop.
    """
    Key length guessing (discrete calculation version)
    """
    # 1) guess length
    L_guess, scored = estimate_key_length(secret, true_ciphertext, max_key_length)
    print(f"Guessed key length = {L_guess}")
    print("Top‑5 candidates (L, score):", sorted(scored, key=lambda x: x[1])[:5])

    # 2) set key_length for gradient attack
    key_length = L_guess
    # rest is identical to your run_test, replacing fixed key_length
    onehot_ct = np.array([one_hot(char_to_index(c)) 
                          for c in true_ciphertext])
    print(f"key_length: {key_length}")
    
    # Initialize candidate key shifts as continuous values from 0-25
    candidate_key_shifts = np.random.uniform(0, 25, key_length)
    # print(f"candidate_key_shifts: {candidate_key_shifts}")
    
    # Set up gradient computation with autograd
    loss_grad = grad(loss_fn)
    loss_history = []
    
    # Gradient descent over n iterations, 200 by default
    for i in range(n_iters):
        current_loss = loss_fn(candidate_key_shifts, true_secret, onehot_true_ciphertext, sigma)
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
        # Longer test cases that require updating the max possible key length
        # ("THISMESSAGEISENCRYPTEDWITHAVERYLONGKEYTOTESTTHEDIFFERENTIABLEAPPROACH",
        #  "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    ]
    
    # BEST RESULTS SO FAR: sigma=2.0, LR=1.0
    # good sigma values to choose: 1.8, 2.0
    # Notes about results:
    # with the above hyperparameters, for the test case with key "GRADIENT" the model correctly obtains the key at ~100 iterations
    # but then obtains "HRADIENT" from ~200th iteration and onward
    # ^ resolved the above with sigma=1.8
    for secret, key in test_cases:
        run_test(secret, key, n_iters=1000, sigma=1.8, learning_rate=1.0)
