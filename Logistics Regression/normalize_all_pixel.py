# GRADED FUNCTION: normalize_per_pixel
def normalize_all_pixels(train_x, test_x):
    """TODO 2: normalize_all_pixels
    This function computes the mean and standard deviation of all pixels and performs data scaling on train_x and test_x using these computed values.

    :param train_x: training images, shape=(num_train, image_height, image_width)
    :param test_x: test images, shape=(num_test, image_height, image_width)
    """
    # The shape of train_mean and train_std should be (1, 1, 1).
    ### START CODE HERE ### (≈4 lines)
    mean = np.sum(train_x) / (train_x.shape[0] * train_x.shape[1] * train_x.shape[2])
    std = (np.sum((train_x - mean)**2) / (train_x.shape[0] * train_x.shape[1] * train_x.shape[2]))**0.5
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    
    ### END CODE HERE ###
    
    return train_x, test_x

### SANITY CHECK
train_x = np.arange(2*2*3).reshape(2,2,3)
assert np.sum(normalize_all_pixels(train_x, train_x)) > 0, "Wrong"
