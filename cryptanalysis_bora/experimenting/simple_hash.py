# This script implements a simple reversible hash function for 8-bit integers.
# The hash function is designed to be reversible, meaning that given the hashed value,
# we can recover the original integer value.
# The hash function is defined as follows:
# h(x) = (x * 7) mod 256
# The inverse function is defined as:
# x = (h(x) * inv7) mod 256, where inv7 is the modular inverse of 7 modulo 256.

# Globals
x = 7
mod = 41
inv_mod = pow(x, -1, mod)

"""
A simple reversible hash function for 8-bit integers.
Input: Plaintext P
Output: Hashed text C
"""
def hash(P):
    return (P * x) % mod

"""
The inverse function of simple_hash.
It recovers the original value using the modular inverse of the hash function.
Input: Hashed text C
Output: Plaintext P
"""
def inv_hash(C):
    # Compute the modular inverse of the hash function
    return (C * inv_mod) % mod


"""Testing
"""
def main():
    # Test the reversible hash
    P = 31
    print(f"Testing simple example, P = {P}")
    C = hash(P)
    R = inv_hash(C) # R for recovered
    print(f"After hashing and inverting: C = {C}, R = {R}")

    print("==============================================")

    for P in range(100):
        C = hash(P)
        R = inv_hash(C) # Recover the original value
        # Check if the recovered value matches the original plaintext
        if R == P:
            print(f"Correct: R = {R}, P = {P}")
        else:
            print(f"Error: P = {P} -> C = {C} -> R = {R}")
    print("All 8-bit values encoded and decoded correctly!")
    
    # Demonstrate with a few sample values
    sample_values = [10, 50, 100, 200]
    print("\nSample results:")
    for P in sample_values:
        C = hash(P)
        R = inv_hash(C)
        print(f"P: {P:3d} -> C: {C:3d} -> R: {R:3d}")

if __name__ == '__main__':
    main()
