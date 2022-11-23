from math import log

# data_test and data_model are matrices of size d * n_test


def anderson_darling(data_model, data_test):
    n_test = len(data_test[0])
    d = len(data_test)
    W = []
    for s in range(d):
        data_model[s].sort()
        U = [
            (sum([int(data_test[s][j] <= data_model[s][i]) for j in range(n_test)]) + 1)
            / (n_test + 2)
            for i in range(n_test)
        ]
        W.append(
            -n_test
            - sum(
                [
                    (2 * (i + 1) - 1) * (log(U[i]) + log(1 - U[n_test - i - 1]))
                    for i in range(n_test)
                ]
            )
            / n_test
        )
    return sum(W) / d
