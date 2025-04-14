# Discrete Substitution cipher

"""Imports
"""
import string
import sub_cipher
from diff_sub_cipher_v1 import char_to_index, index_to_char

"""Globals
"""
ENGLISH_FREQ = [0.082, 0.015, 0.028, 0.043, 0.127, 0.022,
                0.020, 0.061, 0.070, 0.002, 0.008, 0.040,
                0.024, 0.067, 0.075, 0.019, 0.001, 0.060,
                0.063, 0.091, 0.028, 0.010, 0.023, 0.001,
                0.020, 0.001]
ALPHABET = string.ascii_uppercase
ALPHABET_SIZE = 26

# Vigenere Cipher (polyalphabetic cipher)
# e.g. secret = ATTACKATDAWN, key = LEMON
# A T T A C K A T D A W N
# L E M O N L E M O N L E
# Also potential for Chained Viegenere Cipher with multiple keys applied back to back.

def vigenere_encrypt_discrete(secret, key):
    """
    Implements the forward process of polyalphabetic substitution.
    Assumes input only consists of alphabetic chars.
    """
    # Set all text to uppercase and eliminate whitespaces.
    secret = secret.upper().replace(" ", "")
    key = key.upper()

    # Instantiate the output.
    ciphertext = ""
    key_index = 0

    for char in secret:
        # "key_index % len(key)" cycles through the key, since it may be shorter than the secret.
        # Use that value to access the current char from key.
        # Subtract ord('A') to normalize indicies relative to the English alphabet, 0-25.
        # The result of the following computation is how much we are going to shift the char in the secret.
        shift = ord(key[key_index % len(key)]) - ord('A')

        # With the shift calculated, 
        encrypted_char = chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
        ciphertext += encrypted_char
        key_index += 1
    return ciphertext

# Had GPT 4o implement this to help with testing/verification of results
def vigenere_decrypt_discrete(ciphertext, key):
    """
    Given ciphertext and key, performs reverse process of Vigenere encryption and returns the original plaintext.
    """
    plaintext = ""
    key = key.upper()
    for i, c in enumerate(ciphertext):
        shift = char_to_index(key[i % len(key)])
        orig_idx = (char_to_index(c) - shift) % ALPHABET_SIZE
        plaintext += index_to_char(orig_idx)
    return plaintext


# TODO: compare differentiable results with frequency analysis
def freq_analysis():
    """
    Performs classical frequency analysis.
    """
    # With polyalphabnetic ciphers, the frequency analysis approach works best when the length of the key
    # is either known exactly or approximately.
    # Frequency analyses also rely on the frequency at which letters occur in the English language, which
    # is defined globally in this file.

    return


if __name__ == "__main__":
    # Example Vigenere cipher
    secret = "ATTACKATDAWN"
    key = "LEMON"
    print(f"secret = {secret}\nkey = {key}")
    print("Encrypting using Vigenere Cipher...")
    ciphertext = vigenere_encrypt_discrete(secret, key)
    print(f"ciphertext = {ciphertext}")
    if ciphertext != "LXFOPVEFRNHR": print("Something went wrong")
    

