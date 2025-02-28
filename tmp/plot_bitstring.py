import matplotlib.pyplot as plt

def sliding_window_average(data, window_width=3):
    """Calculate the sliding window average of the data."""
    return [sum(data[i:i+window_width]) / window_width for i in range(len(data) - window_width + 1)]

# Generate data
x = list(range(256))  # X axis: index of an 8-bit binary bit string
y = [bin(i).count('1') for i in x]  # Y axis: number of bits in the bitstring

# Calculate sliding window average
window_width = 21
y_avg = sliding_window_average(y, window_width)

# Adjust x to match the length of y_avg and add offset
offset = window_width // 2
x_avg = [i + offset for i in x[:len(y_avg)]]

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot original data
ax1.plot(x, y, marker='o', linestyle='-', color='b', label='Original')

# Highlight consecutive points where the number of ones is the same
start_idx = 0
while start_idx < len(y):
    end_idx = start_idx
    while end_idx < len(y) - 1 and y[end_idx] == y[end_idx + 1]: # and y[end_idx] == 4:
        end_idx += 1
    if end_idx > start_idx:
        ax1.axvspan(start_idx, end_idx, color='yellow', alpha=0.3)
    start_idx = end_idx + 1

# Add labels and title
ax1.set_xlabel('Index of 8-bit Binary Bit String')
ax1.set_ylabel('Number of Bits in Bitstring')
ax1.set_title('Number of Bits in 8-bit Binary Bit Strings')

# Add legend
ax1.legend(loc='upper left')

# Create a secondary y-axis for the histogram
ax2 = ax1.twinx()

# Calculate histogram data
hist_data = [0] * (max(y) + 1)
for value in y:
    hist_data[value] += 1

# Plot histogram
ax2.barh(range(len(hist_data)), hist_data, alpha=0.3, color='g', label='Histogram')

# Add labels for the secondary y-axis
ax2.set_ylabel('Frequency')

# Add legend for the secondary y-axis
ax2.legend(loc='upper right')

# Display the plot
plt.grid(True)
plt.show()
