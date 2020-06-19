# GRADED FUNCTION: add_one
def add_one(x):
    """TODO 4: add_one
    This function add ones as an additional feature for x.

    :param x: input data
    """
    ### START CODE HERE ### (≈1 line)
    x = np.append(x, np.ones((x.shape[0], 1)), axis = 1)
    ### END CODE HERE ###
    return x

### SANITY CHECK
x = np.arange(2*3).reshape(2,3)
assert add_one(x).sum() == 17, "Wrong"
