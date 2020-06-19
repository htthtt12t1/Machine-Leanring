# GRADED FUNCTION
class LogisticClassifier(object):
    def __init__(self, w_shape):
        """__init__
        
        :param w_shape: create w with shape w_shape using normal distribution
        """

        mean = 0
        std = 1
        self.w = np.random.normal(0, np.sqrt(2./np.sum(w_shape)), w_shape)


    def feed_forward(self, x):
        """TODO 5: feed_forward
        This function computes the output of your logistic classification model.
        
        :param x: input
        """
        result = None
        
        ### START CODE HERE ### (≈2 lines)
        z = x.dot(self.w)
        result = 1 / (1 + np.exp(-z))
        ### END CODE HERE ###
        
        return result


    def compute_loss(self, y, y_hat):
        """TODO 6: compute_loss
        Compute the loss using y (label) and y_hat (predicted class).

        :param y:  the label, the actual class of the sample
        :param y_hat: the probabilities that the given sample belong to class 1
        """
        loss = 0
        
        ### START CODE HERE ### (≈2 lines)
        a = y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
        loss = -np.mean(a)
        ### END CODE HERE ###
        
        return loss


    def get_grad(self, x, y, y_hat):
        """TODO 7: get_grad
        Compute and return the gradient of w.

        :param x: input
        :param y: the label, the actual class of the sample data
        :param y_hat: predicted y
        """ 
        w_grad = None
        
        ### START CODE HERE ### (≈2 lines)
        loss = y_hat - y
        w_grad = x.T.dot(loss) / y_hat.shape[0]
        ### END CODE HERE ###
        
        return w_grad


    def update_weight(self, grad, learning_rate):
        """TODO 8: update_weight
        Update w using the computed gradient.

        :param grad: gradient computed from the loss
        :param learning_rate: float, learning rate
        """
        ### START CODE HERE ### (≈1 line)
        self.w = self.w - learning_rate * grad
        ### END CODE HERE ###
        return self.w


    def update_weight_momentum(self, grad, learning_rate, momentum, momentum_rate):
        """TODO 9: update_weight using momentum
        BONUS:[YC1.8]
        Update w using the algorithm with momentum

        :param grad: gradient computed from the loss
        :param learning_rate: float, learning rate
        :param momentum: the array storing momentum for training w, should have the same shape as that of w
        :param momentum_rate: float, how much momentum to reuse after each loop (denoted as gamma in the following section)
        """
        ### START CODE HERE ### (≈3 lines)
        momentum = momentum * momentum_rate + learning_rate * grad 
        self.w = self.w - momentum

        ### END CODE HERE ###
        return self.w


    def numerical_check(self, x, y, grad):
        eps = 0.000005
        w_test0 = np.copy(self.w)
        w_test1 = np.copy(self.w)
        w_test0[2] = w_test0[2] - eps
        w_test1[2] = w_test1[2] + eps

        y_hat0 = np.dot(x, w_test0)
        y_hat0 = 1. / (1. + np.exp(-y_hat0))
        loss0 = self.compute_loss(y, y_hat0) 

        y_hat1 = np.dot(x, w_test1)
        y_hat1 = 1. / (1. + np.exp(-y_hat1))
        loss1 = self.compute_loss(y, y_hat1) 

        numerical_grad = (loss1 - loss0)/(2*eps)
        print(numerical_grad)
        print(grad[2])

# SANITY CHECK
eps = 0.001        
classifer = LogisticClassifier((3,1))
classifer.w = np.arange(3*1).reshape(3,1)
x = np.ones(2*3).reshape(2,3)
y = np.ones(2).reshape(2,1)
y_hat = classifer.feed_forward(x)
assert abs(sum(y_hat) - 1.905) < eps, "Wrong"
loss = classifer.compute_loss(y, y_hat)
assert abs(loss - 0.048) < eps, "Wrong"
grad = classifer.get_grad(x, y, y_hat)
assert abs(sum(grad) + 0.142) < eps, "Wrong"
updateweight = classifer.update_weight(grad, 0.1)
assert abs(sum(updateweight) - 3.014) < eps, "Wrong"
updatemomen = classifer.update_weight_momentum(grad, 0.1, 0.1, 0.1)
assert abs(sum(updatemomen) - 2.998) < eps, "Wrong"
