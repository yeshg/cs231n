import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0] # Number of training images N
  num_classes = W.shape[1] # Number of classes C

  for i in xrange(num_train):
    # Compute vector of scores
    f_i = X[i].dot(W)  # f_i is a vector of class scores for i'th image, ex: [123, 456, 789]
    
    # Normalization trick to avoid numerical instability, per http://cs231n.github.io/linear-classify/#softmax
    f_i -= np.max(f_i) # continuing from above ex, f_i becomes [-666, -333, 0]
    
    # Calculate softmax and compute loss (and add to it, divided later)
    sum_i = np.sum(np.exp(f_i)) # continuing from above ex, sum_i = e^-666 + e^-333 + e^0
    correct_score_i = f_i[ y[i] ] # correct class score
    p_i = np.exp(correct_score_i) / sum_i # softmax probability for the correct class
    Li = -np.log(p_i) 
    loss += Li
    
    # Compute gradient
    # Here we are computing the contribution to the inner sum for a given i.
    for k in range(num_classes):    
        score_k = f_i[k] # the score for the kth class
        p_k = np.exp(score_k)/ sum_i # softmax probability for the kth class
        
        # gradient update for row of dW gradient matrix corresponding to kth class   
        # gradient of loss of ith image with respect to kth score = p_k - 1 if k is y_i class, and p_k for other classes
        # this gives negative loss for the y_i class and positive loss for the other classes
        
        gradient_Li_wrt_k_th_class_score = p_k - (k == y[i]) # Ref http://cs231n.github.io/neural-networks-case-study/
        
        # Backpropogation with chain rule: multiply the gradient on scores d_scores by image data X[i] to update kth column of dW
        dW[:, k] += gradient_Li_wrt_k_th_class_score * X[i]  # dW[:,k] and X[i] shape => [ 3073 * 1 ]
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0] # number of images N

  # Earlier f_i was a 1D vector of scores for ith image, now f is 2D matrix of shape (N, C)
  f = X.dot(W) 

  f -= np.max(f, axis=1, keepdims=True) # find and subtract max of every sample (this is vector of shape (N,) )

  sum_f = np.sum(np.exp(f), axis=1, keepdims=True) # sum_f has shape (N,)
  # print("sum_f shape: ", sum_f.shape)
    
  p = np.exp(f)/sum_f # calculate softmax probabilities for all elements in f. p has shape (N, C)
  # print("p shape: ", p.shape)

  # Compute loss
  softmax_probabilities_for_all_y = p[np.arange(num_train), y] # vector of size (N,) of softmax probabilities for y_i's
  all_loss = -np.log(softmax_probabilities_for_all_y) # vector of size (N,) of Li computations for all N images
  loss = np.sum(all_loss)/num_train # average all these to get the total loss
    
  # Compute gradient
  ind = np.zeros_like(p) # ind is matrix of all 0 value elements of shape p (N, C)
  ind[np.arange(num_train), y] = 1

  gradient_L_wrt_class_score = p - ind # compute gradient of losses with respect to class scores (softmax gradient)
  dW = X.T.dot(gradient_L_wrt_class_score) # compute gradient of W

  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

