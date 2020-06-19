# GRADED FUNCTION: reshape2D
def reshape2D(tensor):
    """TODO 3: reshape_2D
    Reshape our 3D tensors to 2D. A 3D tensor of shape (num_samples, image_height, image_width) must be reshaped into (num_samples, image_height*image_width).
    """
    result = None
    ### START CODE HERE ### (≈1 line)
    result = tensor.reshape(tensor.shape[0], tensor.shape[1] * tensor.shape[2])
    ### END CODE HERE ###
    return result

### SANITY CHECK
tensor = np.arange(2*3*4).reshape(2,3,4)
assert sum(reshape2D(tensor).shape)==14, "Wrong"
