def get_reflected_indices(base_index, window_width, list_length):
    """
    Returns a list of indices into a list, centered at the base_index with a specified window width.
    Indices are reflected if they go out of bounds.
    
    Parameters:
    - base_index (int): The central index of the window.
    - window_width (int): The width of the window.
    - list_length (int): The length of the list.
    
    Returns:
    - List[int]: A list of indices within the specified window.
    """
    radius = window_width // 2
    start_index = base_index - radius
    end_index = base_index + radius + 1

    indices = []
    for i in range(start_index, end_index):
        if i < 0:
            reflected_index = -i - 1
        elif i >= list_length:
            reflected_index = 2 * list_length - i - 1
        else:
            reflected_index = i
        indices.append(reflected_index)
    return indices

def test_get_reflected_indices():
    base_index = 0
    window_width = 2
    list_length = 10

    #expected_indices = [2, 1, 0, 9, 8]
    indices = get_reflected_indices(base_index, window_width, list_length)
    print(indices)
    #assert indices == expected_indices, "Test failed: Incorrect indices returned"

test_get_reflected_indices()