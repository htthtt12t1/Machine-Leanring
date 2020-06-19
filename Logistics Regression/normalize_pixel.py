# GRADED FUNCTION: normalize_per_pixel
def normalize_per_pixel(train_x, test_x):
    """TODO 1: normalize_per_pixel
    This function computes the mean and standard deviation of the pixels located at the same coordinates across and training images
    and performs data scaling on train_x and test_x using these computed values.

    :param train_x: training images, shape=(num_train, image_height, image_width)
    :param test_x: test images, shape=(num_test, image_height, image_width)
    """
    # The shape of train_mean and train_std should be (1, image_height, image_width)
    ### START CODE HERE ### (≈4 lines)
    mean = np.sum(train_x) / train_x.shape[0]
    std = (np.sum((train_x - mean)**2) / (train_x.shape[0]))**0.5
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    print(train_x)
    ### END CODE HERE ###
    
    return train_x, test_x

### SANITY CHECK
train_x = np.arange(2*2*3).reshape(2,2,3)
assert np.sum(normalize_per_pixel(train_x, train_x)) == 0
