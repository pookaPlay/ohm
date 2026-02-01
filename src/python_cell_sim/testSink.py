import numpy as np
#from scipy.optimize import linear_sum_assignment

if __name__ == "__main__":    

    """ 
    This program should implement the Sinkhorn-Knopp algorithm which is known to converge to a permutation matrix 
    How do i get the permutation out of the maatrix? My result looks like: 
    [[0.23618469 0.34889507 0.4148924 ]
     [0.36954004 0.32753357 0.30293676]
     [0.39427528 0.32357138 0.2821709 ]]
    """

    #za = np.zeros([4, 4])
    #z = [[1, 9, 3, 4], [5, 6, 7, 8], [9, 2, 11, 2], [3, 14, 5, 6]]
    z = [[1, 2, 3], [5, -6, 7], [-9, 10, -11]]
    za = np.array(z, dtype=np.float32)
    print(f"Input")
    print(za)
    np.exp(za, za)
    
    print(f"Input normalized")
    print(za)
    
    for iter in range(10):
        print(f"#################################\nIteration {iter}")
        for r in range(za.shape[0]):
            row = za[r, :]
            srow = np.sum(row)
            row = row / srow
            za[r, :] = row        
        print(f"Row normalized")
        print(za)
        zb = za.copy()
        for c in range(za.shape[1]):
            col = za[:, c]
            scol = np.sum(col)
            col = col / scol
            za[:, c] = col
        print(f"Col normalized")
        print(za)

        sumZ = np.sum(np.abs(za-zb))
        if sumZ < 1e-3:
            break        

    # Use the Hungarian algorithm (linear_sum_assignment) to extract the permutation
    # We use -za because linear_sum_assignment finds the minimum cost
    #row_ind, col_ind = linear_sum_assignment(-za)
    #perm = np.zeros_like(za)
    #perm[row_ind, col_ind] = 1.0
    #print(f"Permutation Matrix:\n{perm}")
