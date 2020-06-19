# GRADED FUNCTION
def test(y_hat, test_y, thres=0.5):
    """TODO 10: test
    Compute precision, recall and F1-score based on predicted test values

    :param y_hat: predicted values, output of classifier.feed_forward
    :param test_y: test labels
    """
    
    # Compute test scores using test_y and y_hat

    precision = 0
    recall = 0
    f1 = 0
    ### START CODE HERE ### (≈7 lines)
    TP = np.sum((y_hat > 0.5) * (test_y == 1))
    FP = np.sum((y_hat > 0.5) * (test_y != 1))
    P = np.sum(test_y == 1)
    precision = TP / (TP + FP)
    recall = TP / P
    f1 = 2 * (precision * recall) / (precision + recall)
    ### END CODE HERE ###

    return precision, recall, f1

### SANITY CHECK
y_hat = np.array([0.4, 0.7, 0.8, 0.3, 0.2])
test_y = np.array([0, 1, 1, 0, 0])
assert sum(test(y_hat, test_y)) == 3, "Wrong"
