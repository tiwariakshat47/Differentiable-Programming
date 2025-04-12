Nomenclature:
- plaintext <-> secret
- ciphertext <-> ciphertext (?)
- attempt at recovering plaintext <-> tecret
- alphabet <-> key
- encryption <-> forward process
- decryption <-> reverse process

Notes:
- monoalphabetic substitution
    - each letter always maps to same cipher letter
    - e.g. you have a key which tells you how the letters are shifted
    - this is very trivial, Casear Cipher for example
- polyalphabetic substitution (Vigenere cipher)
    - uses multiple substitution alphabets (see example below)
    - secret = ATTACKATDAWN, key = LEMON
    - A T T A C K A T D A W N
    - L E M O N L E M O N L E
    - L X F O P V E F R N H R
    - e.g. vigenere cipher
- frequency analysis is applicable to both, but poly is more durable
- can apply multiple alphabets to make cipher stronger


