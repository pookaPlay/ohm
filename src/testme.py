import matplotlib.pyplot as plt

def plot_bitstring_ones(K):
    # Generate all possible bitstrings of length K
    bitstrings = [bin(i)[2:].zfill(K) for i in range(2**K)]
    
    # Calculate the number of 1's in each bitstring
    ones_count = [bitstring.count('1') for bitstring in bitstrings]
    
    # Calculate the consecutive differences in the number of 1's
    ones_diff = [abs(ones_count[i+1] - ones_count[i]) for i in range(len(ones_count)-1)]
    
    # Plot the consecutive differences
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 2**K), ones_diff, marker='o', color='r')
    plt.xlabel('Index of Bitstring')
    plt.ylabel('Consecutive Difference in Number of 1\'s')
    plt.title(f'Consecutive Difference in Number of 1\'s in {K}-bit Bitstrings')
    plt.grid(True)
    plt.show()
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(range(2**K), ones_count, marker='o')
    plt.xlabel('Index of Bitstring')
    plt.ylabel('Number of 1\'s in Bitstring')
    plt.title(f'Number of 1\'s in {K}-bit Bitstrings')
    plt.grid(True)
    plt.show()

# Example usage
plot_bitstring_ones(8)
