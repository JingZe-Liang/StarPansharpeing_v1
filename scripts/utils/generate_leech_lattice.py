import numpy as np
import torch
import os
from itertools import combinations

def get_golay_generator():
    """
    Returns the generator matrix for the extended binary Golay code G24.
    The matrix is in the form [I | P].
    """
    # Standard P matrix for G24 (12x12)
    # This is based on the adjacency matrix of the icosahedron or other standard constructions
    P = np.array([
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
        [1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1],
        [1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0],
        [1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0],
        [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0],
        [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0],
        [1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0]
    ], dtype=np.int8)
    
    G = np.hstack([np.eye(12, dtype=np.int8), P])
    return G

def generate_golay_codewords():
    """Generates all 4096 codewords of G24."""
    G = get_golay_generator()
    codewords = []
    for i in range(4096):
        bits = np.array([(i >> j) & 1 for j in range(12)], dtype=np.int8)
        cw = (bits @ G) % 2
        codewords.append(cw)
    return np.array(codewords)

def generate_leech_minimal_vectors():
    """
    Generates the 196,560 minimal vectors of the Leech lattice.
    Norm squared is 32.
    """
    codewords = generate_golay_codewords()
    
    # Find octads (weight 8 codewords)
    weights = codewords.sum(axis=1)
    octads = codewords[weights == 8]
    assert len(octads) == 759
    
    vectors = []
    
    print("Generating Type 1: (±4, ±4, 0^22)...")
    # Type 1: (±4, ±4, 0^22)
    # 2^2 * binom(24, 2) = 4 * 276 = 1104
    for i, j in combinations(range(24), 2):
        for s1 in [4, -4]:
            for s2 in [4, -4]:
                v = np.zeros(24, dtype=np.float32)
                v[i] = s1
                v[j] = s2
                vectors.append(v)
                
    print("Generating Type 2: (±2^8, 0^16)...")
    # Type 2: (±2^8, 0^16)
    # 759 * 2^7 = 97152
    # Signs must have even number of minuses
    for octad in octads:
        indices = np.where(octad == 1)[0]
        for i in range(128): # 2^7 combinations
            bits = [(i >> j) & 1 for j in range(7)]
            # Last bit is parity bit to ensure even number of minuses
            bits.append(sum(bits) % 2)
            
            v = np.zeros(24, dtype=np.float32)
            for idx, bit in zip(indices, bits):
                v[idx] = 2 if bit == 0 else -2
            vectors.append(v)

    print("Generating Type 3: (∓3, ±1^23)...")
    # Type 3: (∓3, ±1^23)
    # 24 * 2^12 = 98304
    # v_i = (-1)^{c_i} * 3, v_j = (-1)^{c_j} * 1
    # But wait, the standard construction is:
    # v = (x_1, ..., x_24) where x_i = (-1)^{c_i} - 4*delta_{i,k}
    # This gives (-3, 1, 1, ...) if c_i=0, or (3, -1, -1, ...) if c_i=1
    for k in range(24):
        for cw in codewords:
            # We only need to consider one parity of the codeword or something?
            # Actually, for each k, there are 2^12 codewords.
            # But (cw) and (1-cw) might produce the same or negative vectors.
            # The kissing number is 196560. 
            # 1104 + 97152 + 98304 = 196560.
            # Type 3 is 98304. 98304 / 24 = 4096.
            # So for each position k, we use all 4096 codewords.
            v = (1 - 2 * cw).astype(np.float32)
            v[k] -= 4 * v[k] # If v[k] was 1, it becomes -3. If -1, it becomes 3.
            vectors.append(v)

    vectors = np.array(vectors)
    print(f"Total vectors generated: {len(vectors)}")
    assert len(vectors) == 196560
    return vectors

def main():
    output_dir = "/home/user/zihancao/Project/hyperspectral-1d-tokenizer/src/stage1/discretization/InfinityCC/cache"
    os.makedirs(output_dir, exist_ok=True)
    
    vectors = generate_leech_minimal_vectors()
    
    # Normalize as per the project's requirement (divide by sqrt(32))
    # The code says: leech_lattices = np.load(...) * math.sqrt(32)
    # and then converts to long. This means the saved values are v / sqrt(32).
    normalized_vectors = vectors / np.sqrt(32)
    
    output_path = os.path.join(output_dir, "leech_lattices_normalized.npy")
    np.save(output_path, normalized_vectors)
    print(f"Saved normalized Leech lattice vectors to {output_path}")

if __name__ == "__main__":
    main()
