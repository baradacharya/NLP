import numpy as np

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]
    y = []
    R = np.zeros((N,L)) #Score
    path = np.zeros((N-1,L)) #Store path for backtracking (i<-j)

    for j in range(L):
        R[0][j] = start_scores[j] +  emission_scores[0][j]

    for i in range(1,N):
        for j in range(L):
            max_temp = -np.inf
            for k in range(L):
                temp = R[i-1][k] + emission_scores[i][j] + trans_scores[k][j]
                if (max_temp <= temp):
                    max_temp = temp
                    R[i][j] = max_temp
                    path[i-1][j] = k #Trans from j to k


    _max = -np.inf
    for k in range(L):
        temp = R[N-1][k] + end_scores[k]
        if (_max <= temp):
            _max = temp
            index = k
    y.append(index)
    if N > 1:
        for i in range(N-2,-1,-1):
            y.append(int(path[i][y[-1]]))
    y.reverse()

    return (_max, y)