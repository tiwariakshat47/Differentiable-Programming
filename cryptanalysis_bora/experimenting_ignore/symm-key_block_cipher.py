# Simple Two-Round SPN Block Cipher Example

# We use a 4-bit S-box for substitution. This S-box maps each 4-bit nibble to another 4-bit value.
# These values are taken from a common toy S-box used in cryptanalysis examples.
SBOX = [
    0xE, 0x4, 0xD, 0x1,
    0x2, 0xF, 0xB, 0x8,
    0x3, 0xA, 0x6, 0xC,
    0x5, 0x9, 0x0, 0x7
]

# Build the inverse S-box for decryption.
INV_SBOX = [0] * 16
for i, val in enumerate(SBOX):
    INV_SBOX[val] = i

# Permutation mapping on the nibbles.
# We interpret a 16-bit block as consisting of four 4-bit nibbles: [N0, N1, N2, N3],
# where N0 is the most significant nibble.
# Here, we define a permutation that reorders the nibbles.
PERMUTATION = [2, 0, 3, 1]  # After permutation: new order: N2, N0, N3, N1

# The inverse permutation mapping for decryption.
INV_PERMUTATION = [0] * 4
for i, p in enumerate(PERMUTATION):
    INV_PERMUTATION[p] = i

def substitute(block, sbox):
    """
    Substitute each 4-bit nibble of the 16-bit block using the given S-box.
    """
    output = 0
    # Process 4 nibbles (16 bits total); nibble 0 is most significant.
    for i in range(4):
        # Extract nibble: shift right by (3-i)*4 and mask with 0xF.
        nibble = (block >> ((3 - i) * 4)) & 0xF
        # Substitute using the S-box.
        sub_nibble = sbox[nibble]
        # Place the substituted nibble back into its position.
        output |= (sub_nibble << ((3 - i) * 4))
    return output

def permute(block, permutation):
    """
    Permute the 4 nibbles of the 16-bit block according to the given permutation mapping.
    """
    # First, extract the 4 nibbles into a list.
    nibbles = []
    for i in range(4):
        nibble = (block >> ((3 - i) * 4)) & 0xF
        nibbles.append(nibble)
    # Rearrange the nibbles according to the permutation mapping.
    permuted_nibbles = [0] * 4
    for i, new_pos in enumerate(permutation):
        permuted_nibbles[new_pos] = nibbles[i]
    # Reconstruct the 16-bit block.
    output = 0
    for i in range(4):
        output |= (permuted_nibbles[i] << ((3 - i) * 4))
    return output

def encrypt_block(plaintext, key1, key2):
    """
    Encrypt a 16-bit plaintext block with two rounds.
    
    Round 1: XOR with key1, substitute, then permute.
    Round 2: XOR with key2, then substitute.
    
    Returns a 16-bit ciphertext block.
    """
    # Round 1: Key mixing, substitution, and permutation.
    state = plaintext ^ key1
    state = substitute(state, SBOX)
    state = permute(state, PERMUTATION)
    
    # Round 2: Key mixing and substitution.
    state = state ^ key2
    state = substitute(state, SBOX)
    
    return state

def decrypt_block(ciphertext, key1, key2):
    """
    Decrypt a 16-bit ciphertext block by inverting the operations.
    
    Inverse round 2: Inverse substitution, XOR with key2.
    Inverse round 1: Inverse permutation, inverse substitution, XOR with key1.
    
    Returns a 16-bit plaintext block.
    """
    # Inverse Round 2:
    state = substitute(ciphertext, INV_SBOX)
    state = state ^ key2
    
    # Inverse Round 1:
    state = permute(state, INV_PERMUTATION)
    state = substitute(state, INV_SBOX)
    state = state ^ key1
    
    return state

# === EXAMPLE USAGE ===

if __name__ == '__main__':
    # Example 16-bit plaintext block, keys
    plaintext = 0x1234          # In hexadecimal (16-bit)
    key1 = 0xAAAA              # Round 1 key (example 16-bit)
    key2 = 0x5555              # Round 2 key (example 16-bit)
    
    # Encrypt the plaintext
    ciphertext = encrypt_block(plaintext, key1, key2)
    
    # Decrypt the ciphertext
    recovered_plaintext = decrypt_block(ciphertext, key1, key2)
    
    # Print the results
    print("=== Simple Two-Round SPN Block Cipher ===")
    print(f"Plaintext:          0x{plaintext:04X}")
    print(f"Round 1 Key:        0x{key1:04X}")
    print(f"Round 2 Key:        0x{key2:04X}")
    print(f"Ciphertext:         0x{ciphertext:04X}")
    print(f"Recovered Plaintext:0x{recovered_plaintext:04X}")
    
    # Verify correct decryption
    if plaintext == recovered_plaintext:
        print("✅ Decryption successful!")
    else:
        print("❌ Decryption failed!")
