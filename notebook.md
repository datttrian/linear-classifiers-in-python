# KNN classification

In this exercise you'll explore a subset of the [Large Movie Review
Dataset](https://ai.stanford.edu/~amaas/data/sentiment/). The variables
`X_train`, `X_test`, `y_train`, and `y_test` are already loaded into the
environment. The `X` variables contain features based on the words in
the movie reviews, and the `y` variables contain labels for whether the
review sentiment is positive (+1) or negative (-1).

*This course touches on a lot of concepts you may have forgotten, so if
you ever need a quick refresher, download the [scikit-learn Cheat
Sheet](https://www.datacamp.com/cheat-sheet/scikit-learn-cheat-sheet-python-machine-learning)
and keep it handy!*

**Instructions**

- Create a KNN model with default hyperparameters.
- Fit the model.
- Print out the prediction for the test example 0.

**Answer**

# Comparing models

Compare k nearest neighbors classifiers with k=1 and k=5 on the
handwritten digits data set, which is already loaded into the variables
`X_train`, `y_train`, `X_test`, and `y_test`. You can set k with the
`n_neighbors` parameter when creating the `KNeighborsClassifier` object,
which is also already imported into the environment.

Which model has a higher test accuracy?

**InstructionsAnswer**

# Running LogisticRegression and SVC

In this exercise, you'll apply logistic regression and a support vector
machine to classify images of handwritten digits.

**Instructions**

- Apply logistic regression and SVM (using `SVC()`) to the handwritten
  digits data set using the provided train/validation split.
- For each classifier, print out the training and validation accuracy.

**Answer**

# Sentiment analysis for movie reviews

In this exercise you'll explore the probabilities outputted by logistic
regression on a subset of the [Large Movie Review
Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

The variables `X` and `y` are already loaded into the environment. `X`
contains features based on the number of times words appear in the movie
reviews, and `y` contains labels for whether the review sentiment is
positive (+1) or negative (-1).

**Instructions**

- Train a logistic regression model on the movie review data.
- Predict the probabilities of negative vs. positive for the two given
  reviews.
- Feel free to write your own reviews and get probabilities for those
  too!

**Answer**

# Visualizing decision boundaries

In this exercise, you'll visualize the decision boundaries of various
classifier types.

A subset of `scikit-learn`'s built-in `wine` dataset is already loaded
into `X`, along with binary labels in `y`.

**Instructions**

- Create the following classifier objects with default hyperparameters:
  `LogisticRegression`, `LinearSVC`, `SVC`, `KNeighborsClassifier`.
- Fit each of the classifiers on the provided data using a `for` loop.
- Call the `plot_4_classifers()` function (similar to the code
  [here](https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html)),
  passing in `X`, `y`, and a list containing the four classifiers.

**Answer**

# Changing the model coefficients

When you call `fit` with scikit-learn, the logistic regression
coefficients are automatically learned from your dataset. In this
exercise you will explore how the decision boundary is represented by
the coefficients. To do so, you will change the coefficients manually
(instead of with `fit`), and visualize the resulting classifiers.

A 2D dataset is already loaded into the environment as `X` and `y`,
along with a linear classifier object `model`.

**Instructions**

- Set the two coefficients and the intercept to various values and
  observe the resulting decision boundaries.
- Try to build up a sense of how the coefficients relate to the decision
  boundary.
- Set the coefficients and intercept such that the model makes no errors
  on the given training data.

**Answer**

# Minimizing a loss function

In this exercise you'll implement linear regression "from scratch" using
`scipy.optimize.minimize`.

We'll train a model on the Boston housing price data set, which is
already loaded into the variables `X` and `y`. For simplicity, we won't
include an intercept in our regression model.

**Instructions**

- Fill in the loss function for least squares linear regression.
- Print out the coefficients from fitting sklearn's `LinearRegression`.

**Answer**

# Comparing the logistic and hinge losses

In this exercise you'll create a plot of the logistic and hinge losses
using their mathematical expressions, which are provided to you.

The loss function diagram from the video is shown on the right.

**Instructions**

- Evaluate the `log_loss()` and `hinge_loss()` functions **at the grid
  points** so that they are plotted.

**Answer**

# Implementing logistic regression

This is very similar to the earlier exercise where you implemented
linear regression "from scratch" using `scipy.optimize.minimize`.
However, this time we'll minimize the logistic loss and compare with
scikit-learn's `LogisticRegression` (we've set `C` to a large value to
disable regularization; more on this in Chapter 3!).

The `log_loss()` function from the previous exercise is already defined
in your environment, and the `sklearn` breast cancer prediction dataset
(first 10 features, standardized) is loaded into the variables `X` and
`y`.

**Instructions**

- Input the number of training examples into `range()`.
- Fill in the loss function for logistic regression.
- Compare the coefficients to sklearn's `LogisticRegression`.

**Answer**

# Regularized logistic regression

In Chapter 1, you used logistic regression on the handwritten digits
data set. Here, we'll explore the effect of L2 regularization.

The handwritten digits dataset is already loaded, split, and stored in
the variables `X_train`, `y_train`, `X_valid`, and `y_valid`. The
variables `train_errs` and `valid_errs` are already initialized as empty
lists.

**Instructions**

- Loop over the different values of `C_value`, creating and fitting a
  `LogisticRegression` model each time.
- Save the error on the training set and the validation set for each
  model.
- Create a plot of the training and testing error as a function of the
  regularization parameter, `C`.
- Looking at the plot, what's the best value of `C`?

**Answer**

# Logistic regression and feature selection

In this exercise we'll perform feature selection on the movie review
sentiment data set using L1 regularization. The features and targets are
already loaded for you in `X_train` and `y_train`.

We'll search for the best value of `C` using scikit-learn's
`GridSearchCV()`, which was covered in the prerequisite course.

**Instructions**

- Instantiate a logistic regression object that uses L1 regularization.
- Find the value of `C` that minimizes cross-validation error.
- Print out the number of selected features for this value of `C`.

**Answer**

# Identifying the most positive and negative words

In this exercise we'll try to interpret the coefficients of a logistic
regression fit on the movie review sentiment dataset. The model object
is already instantiated and fit for you in the variable `lr`.

In addition, the words corresponding to the different features are
loaded into the variable `vocab`. For example, since `vocab[100]` is
"think", that means feature 100 corresponds to the number of times the
word "think" appeared in that movie review.

**Instructions**

- Find the words corresponding to the 5 largest coefficients.
- Find the words corresponding to the 5 smallest coefficients.

**Answer**

# Regularization and probabilities

In this exercise, you will observe the effects of changing the
regularization strength on the predicted probabilities.

A 2D binary classification dataset is already loaded into the
environment as `X` and `y`.

**Instructions**

- Compute the maximum predicted probability.
- Run the provided code and take a look at the plot.

<!-- -->

- Create a model with `C=0.1` and examine how the plot and probabilities
  change.

**Answer**

# Visualizing easy and difficult examples

In this exercise, you'll visualize the examples that the logistic
regression model is most and least confident about by looking at the
largest and smallest predicted probabilities.

The handwritten digits dataset is already loaded into the variables `X`
and `y`. The `show_digit` function takes in an integer index and plots
the corresponding image, with some extra information displayed above the
image.

**Instructions**

- Fill in the first blank with the *index* of the digit that the model
  is most confident about.
- Fill in the second blank with the *index* of the digit that the model
  is least confident about.
- Observe the images: do you agree that the first one is less ambiguous
  than the second?

**Answer**

# Fitting multi-class logistic regression

In this exercise, you'll fit the two types of multi-class logistic
regression, one-vs-rest and softmax/multinomial, on the handwritten
digits data set and compare the results. The handwritten digits dataset
is already loaded and split into `X_train`, `y_train`, `X_test`, and
`y_test`.

**Instructions**

- Fit a one-vs-rest logistic regression classifier by setting the
  `multi_class` parameter and report the results.
- Fit a multinomial logistic regression classifier by setting the
  `multi_class` parameter and report the results.

**Answer**

# Visualizing multi-class logistic regression

In this exercise we'll continue with the two types of multi-class
logistic regression, but on a toy 2D data set specifically designed to
break the one-vs-rest scheme.

The data set is loaded into `X_train` and `y_train`. The two logistic
regression objects,`lr_mn` and `lr_ovr`, are already instantiated (with
`C=100`), fit, and plotted.

Notice that `lr_ovr` never predicts the dark blue class… yikes! Let's
explore why this happens by plotting one of the binary classifiers that
it's using behind the scenes.

**Instructions**

- Create a new logistic regression object (also with `C=100`) to be used
  for binary classification.
- Visualize this binary classifier with `plot_classifier`… does it look
  reasonable?

**Answer**

# One-vs-rest SVM

As motivation for the next and final chapter on support vector machines,
we'll repeat the previous exercise with a non-linear SVM. Once again,
the data is loaded into `X_train`, `y_train`, `X_test`, and `y_test` .

Instead of using `LinearSVC`, we'll now use scikit-learn's `SVC` object,
which is a non-linear "kernel" SVM (much more on what this means in
Chapter 4!). Again, your task is to create a plot of the binary
classifier for class 1 vs. rest.

**Instructions**

- Fit an `SVC` called `svm_class_1` to predict class 1 vs. other
  classes.
- Plot this classifier.

**Answer**

# Effect of removing examples

Support vectors are defined as training examples that influence the
decision boundary. In this exercise, you'll observe this behavior by
removing non support vectors from the training set.

The wine quality dataset is already loaded into `X` and `y` (first two
features only). (Note: we specify `lims` in `plot_classifier()` so that
the two plots are forced to use the same axis limits and can be compared
directly.)

**Instructions**

- Train a linear SVM on the whole data set.
- Create a new data set containing only the support vectors.
- Train a new linear SVM on the smaller data set.

**Answer**

# GridSearchCV warm-up

In the video we saw that increasing the RBF kernel hyperparameter
`gamma` increases training accuracy. In this exercise we'll search for
the `gamma` that maximizes cross-validation accuracy using
scikit-learn's `GridSearchCV`. A binary version of the handwritten
digits dataset, in which you're just trying to predict whether or not an
image is a "2", is already loaded into the variables `X` and `y`.

**Instructions**

- Create a `GridSearchCV` object.
- Call the `fit()` method to select the best value of `gamma` based on
  cross-validation accuracy.

**Answer**

# Jointly tuning gamma and C with GridSearchCV

In the previous exercise the best value of `gamma` was 0.001 using the
default value of `C`, which is 1. In this exercise you'll search for the
best combination of `C` and `gamma` using `GridSearchCV`.

As in the previous exercise, the 2-vs-not-2 digits dataset is already
loaded, but this time it's split into the variables `X_train`,
`y_train`, `X_test`, and `y_test`. Even though cross-validation already
splits the training set into parts, it's often a good idea to hold out a
separate test set to make sure the cross-validation results are
sensible.

**Instructions**

- Run `GridSearchCV` to find the best hyperparameters using the training
  set.
- Print the best values of the parameters.
- Print out the accuracy on the test set, which was not used during the
  cross-validation procedure.

**Answer**

# Using SGDClassifier

In this final coding exercise, you'll do a hyperparameter search over
the regularization strength and the loss (logistic regression vs. linear
SVM) using `SGDClassifier()`.

**Instructions**

- Instantiate an `SGDClassifier` instance with `random_state=0`.
- Search over the regularization strength and the `hinge` vs. `log_loss`
  losses.

**Answer**
