from functools import reduce

def abs_kendall_error(data_model, data_test):
    n_test = len(data_test[0])
    R_test = [
        sum([intersect(data_test, i, j) for j in range(n_test) if j != i])
        / (n_test - 1)
        for i in range(n_test)
    ]
    R_model = [
        sum([intersect(data_model, i, j) for j in range(n_test) if j != i])
        / (n_test - 1)
        for i in range(n_test)
    ]
    R_test.sort()
    R_model.sort()
    result = 0
    for i in range(n_test):
        result += abs(R_test[i] - R_model[i])
    return result / n_test


def intersect(data, i, j):
    return reduce(
        lambda x, y: x * y, [int(station[j] < station[i]) for station in data]
    )
