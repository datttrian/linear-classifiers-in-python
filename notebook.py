# added/edited
import numpy as np
from sklearn.datasets import load_svmlight_file

X_train, y_train = load_svmlight_file("aclImdb/train/labeledBow.feat")
X_test, y_test = load_svmlight_file("aclImdb/test/labeledBow.feat")
y_train = np.where(y_train < 5, -1.0, 1.0)
y_test = np.where(y_test < 5, -1.0, 1.0)
X_train = X_train[:, : X_test.shape[1]]


from sklearn.neighbors import KNeighborsClassifier

# Create and fit the model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Predict on the test features, print the results
pred = knn.predict(X_test)[0]
print("Prediction for test example 0:", pred)


# added/edited
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


from sklearn import datasets

digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

# Apply logistic regression and print scores
lr = LogisticRegression()
lr.fit(X_train, y_train)
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))

# Apply SVM and print scores
svm = SVC()
svm.fit(X_train, y_train)
print(svm.score(X_train, y_train))
print(svm.score(X_test, y_test))


# added/edited
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

vocabulary = (
    pd.read_csv("aclImdb/imdb.vocab", header=None, names=["word"])
    .drop_duplicates()["word"]
    .values
)

X, y = load_svmlight_file("aclImdb/train/labeledBow.feat")
y = np.where(y < 5, -1.0, 1.0)
X = X[:, : len(vocabulary)]
vectorizer = CountVectorizer(vocabulary=vocabulary)


def get_features(review):
    return vectorizer.transform([review])


# Instantiate logistic regression and train
lr = LogisticRegression()
lr.fit(X, y)

# Predict sentiment for a glowing review
review1 = "LOVED IT! This movie was amazing. Top 10 this year."
review1_features = get_features(review1)
print("Review:", review1)
print("Probability of positive review:", lr.predict_proba(review1_features)[0, 1])

# Predict sentiment for a poor review
review2 = "Total junk! I'll never watch a film by that director again, no matter how good the reviews."
review2_features = get_features(review2)
print("Review:", review2)
print("Probability of positive review:", lr.predict_proba(review2_features)[0, 1])


# added/edited
import matplotlib.pyplot as plt

X = np.array([[11.45, 2.4], [13.62, 4.95], [13.88, 1.89], [12.42, 2.55], [12.81, 2.31], [12.58, 1.29], [13.83, 1.57], [13.07, 1.5], [12.7, 3.55], [13.77, 1.9], [12.84, 2.96], [12.37, 1.63], [13.51, 1.8], [13.87, 1.9], [12.08, 1.39], [13.58, 1.66], [13.08, 3.9], [11.79, 2.13], [12.45, 3.03], [13.68, 1.83], [13.52, 3.17], [13.5, 3.12], [12.87, 4.61], [14.02, 1.68], [12.29, 3.17], [12.08, 1.13], [12.7, 3.87], [11.03, 1.51], [13.32, 3.24], [14.13, 4.1], [13.49, 1.66], [11.84, 2.89], [13.05, 2.05], [12.72, 1.81], [12.82, 3.37], [13.4, 4.6], [14.22, 3.99], [13.72, 1.43], [12.93, 2.81], [11.64, 2.06], [12.29, 1.61], [11.65, 1.67], [13.28, 1.64], [12.93, 3.8], [13.86, 1.35], [11.82, 1.72], [12.37, 1.17], [12.42, 1.61], [13.9, 1.68], [14.16, 2.51]])
y = np.array([True, True, False, True, True, True, False, False, True, False, True, True, False, False, True, False, True, True, True, False, True, True, True, False, True, True, True, True, True, True, True, True, False, True, True, True, False, False, True, True, True, True, False, False, False, True, True, True, False, True])


# added/edited
def make_meshgrid(x, y, h=0.02, lims=None):
    """Create a mesh of points to plot in

    Parameters
    ----------
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional

    Returns
    -------
        xx, yy : ndarray
    """

    if lims is None:
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
    else:
        x_min, x_max, y_min, y_max = lims
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, proba=False, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
        ax: matplotlib axes object
        clf: a classifier
        xx: meshgrid ndarray
        yy: meshgrid ndarray
        params: dictionary of params to pass to contourf, optional
    """
    if proba:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, -1]
        Z = Z.reshape(xx.shape)
        out = ax.imshow(
            Z,
            extent=(np.min(xx), np.max(xx), np.min(yy), np.max(yy)),
            origin="lower",
            vmin=0,
            vmax=1,
            **params
        )
        ax.contour(xx, yy, Z, levels=[0.5])
    else:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_classifier(X, y, clf, ax=None, ticks=False, proba=False, lims=None):
    # assumes classifier "clf" is already fit
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1, lims=lims)

    if ax is None:
        plt.figure()
        ax = plt.gca()
        show = True
    else:
        show = False

    # can abstract some of this into a higher-level function for learners to call
    cs = plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8, proba=proba)
    if proba:
        cbar = plt.colorbar(cs)
        cbar.ax.set_ylabel(
            "probability of red $\Delta$ class", fontsize=20, rotation=270, labelpad=30
        )
        cbar.ax.tick_params(labelsize=14)
        # ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=30, edgecolors=\'k\', linewidth=1)
    labels = np.unique(y)
    if len(labels) == 2:
        ax.scatter(
            X0[y == labels[0]],
            X1[y == labels[0]],
            cmap=plt.cm.coolwarm,
            s=60,
            c="b",
            marker="o",
            edgecolors="k",
        )
        ax.scatter(
            X0[y == labels[1]],
            X1[y == labels[1]],
            cmap=plt.cm.coolwarm,
            s=60,
            c="r",
            marker="^",
            edgecolors="k",
        )
    else:
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=50, edgecolors="k", linewidth=1)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    #     ax.set_xlabel(data.feature_names[0])
    #     ax.set_ylabel(data.feature_names[1])
    if ticks:
        ax.set_xticks(())
        ax.set_yticks(())
        #     ax.set_title(title)
    if show:
        plt.show()
    else:
        return ax


def plot_4_classifiers(X, y, clfs):
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    for clf, ax, title in zip(clfs, sub.flatten(), ("(1)", "(2)", "(3)", "(4)")):
        # clf.fit(X, y)
        plot_classifier(X, y, clf, ax, ticks=True)
        ax.set_title(title)


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier

# Define the classifiers
classifiers = [LogisticRegression(), LinearSVC(), SVC(), KNeighborsClassifier()]

# Fit the classifiers
for c in classifiers:
    c.fit(X, y)

# Plot the classifiers
plot_4_classifiers(X, y, classifiers)
plt.show()


# added/edited
from sklearn.linear_model import LogisticRegression

X = np.array(
    [
        [1.78862847, 0.43650985],
        [0.09649747, -1.8634927],
        [-0.2773882, -0.35475898],
        [-3.08274148, 2.37299932],
        [-3.04381817, 2.52278197],
        [-1.31386475, 0.88462238],
        [-2.11868196, 4.70957306],
        [-2.94996636, 2.59532259],
        [-3.54535995, 1.45352268],
        [0.98236743, -1.10106763],
        [-1.18504653, -0.2056499],
        [-1.51385164, 3.23671627],
        [-4.02378514, 2.2870068],
        [0.62524497, -0.16051336],
        [-3.76883635, 2.76996928],
        [0.74505627, 1.97611078],
        [-1.24412333, -0.62641691],
        [-0.80376609, -2.41908317],
        [-0.92379202, -1.02387576],
        [1.12397796, -0.13191423],
    ]
)
y = np.array([-1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1])
model = LogisticRegression()
model.fit(X, y)


# Set the coefficients
model.coef_ = np.array([[0, 1]])
model.intercept_ = np.array([0])

# Plot the data and decision boundary
plot_classifier(X, y, model)

# Print the number of errors
num_err = np.sum(y != model.predict(X))
print("Number of errors:", num_err)


# added/edited
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

boston = np.genfromtxt("boston.csv", delimiter=",", skip_header=1)
X = boston[:, :-1]
y = boston[:, -1]


# The squared error, summed over training examples
def my_loss(w):
    s = 0
    for i in range(y.size):
        # Get the true and predicted target values for example 'i'
        y_i_true = y[i]
        y_i_pred = w @ X[i]
        s = s + (y_i_true - y_i_pred) ** 2
    return s


# Returns the w that makes my_loss(w) smallest
w_fit = minimize(my_loss, X[0]).x
print(w_fit)

# Compare with scikit-learn's LinearRegression coefficients
lr = LinearRegression(fit_intercept=False).fit(X, y)
print(lr.coef_)


# Mathematical functions for logistic and hinge losses
def log_loss(raw_model_output):
    return np.log(1 + np.exp(-raw_model_output))


def hinge_loss(raw_model_output):
    return np.maximum(0, 1 - raw_model_output)


# Create a grid of values and plot
grid = np.linspace(-2, 2, 1000)
plt.plot(grid, log_loss(grid), label="logistic")
plt.plot(grid, hinge_loss(grid), label="hinge")
plt.legend()
plt.show()


# added/edited
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

breast = np.genfromtxt("breast.csv", delimiter=",", skip_header=1)
X = breast[:, :-1]
y = breast[:, -1]


# The logistic loss, summed over training examples
def my_loss(w):
    s = 0
    for i in range(y.size):
        raw_model_output = w @ X[i]
        s = s + log_loss(raw_model_output * y[i])
    return s


# Returns the w that makes my_loss(w) smallest
w_fit = minimize(my_loss, X[0]).x
print(w_fit)

# Compare with scikit-learn's LogisticRegression
lr = LogisticRegression(fit_intercept=False, C=1000000).fit(X, y)
print(lr.coef_)


# added/edited
X_train, X_valid, y_train, y_valid = train_test_split(
    digits.data, digits.target, stratify=digits.target, random_state=42
)
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]


# Train and validaton errors initialized as empty list
train_errs = list()
valid_errs = list()

# Loop over values of C_value
for C_value in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    # Create LogisticRegression object and fit
    lr = LogisticRegression(C=C_value)
    lr.fit(X_train, y_train)

    # Evaluate error rates and append to lists
    train_errs.append(1.0 - lr.score(X_train, y_train))
    valid_errs.append(1.0 - lr.score(X_valid, y_valid))

# Plot results
plt.semilogx(C_values, train_errs, C_values, valid_errs)
plt.legend(("train", "validation"))
plt.show()


# added/edited
from sklearn.model_selection import GridSearchCV

X_train, y_train = load_svmlight_file("aclImdb/train/labeledBow.feat")
y_train = np.where(y_train < 5, -1.0, 1.0)


# Specify L1 regularization
lr = LogisticRegression(solver="liblinear", penalty="l1")

# Instantiate the GridSearchCV object and run the search
searcher = GridSearchCV(lr, {"C": [0.001, 0.01, 0.1, 1, 10]})
searcher.fit(X_train, y_train)

# Report the best parameters
print("Best CV params", searcher.best_params_)

# Find the number of nonzero coefficients (selected features)
best_lr = searcher.best_estimator_
coefs = best_lr.coef_
print("Total number of features:", coefs.size)
print("Number of selected features:", np.count_nonzero(coefs))


# added/edited
vocab = (
    pd.read_csv("aclImdb/imdb.vocab", header=None, names=["word"])
    .drop_duplicates()["word"]
    .values
)
lr = best_lr


# Get the indices of the sorted cofficients
inds_ascending = np.argsort(lr.coef_.flatten())
inds_descending = inds_ascending[::-1]

# Print the most positive words
print("Most positive words: ", end="")
for i in range(5):
    print(vocab[inds_descending[i]], end=", ")
print("\n")

# Print most negative words
print("Most negative words: ", end="")
for i in range(5):
    print(vocab[inds_ascending[i]], end=", ")
print("\n")


# added/edited
binary = pd.read_csv("binary.csv", header=None).to_numpy()
X = binary[:, :-1]
y = binary[:, -1]


# Set the regularization strength
model = LogisticRegression(C=1)

# Fit and plot
model.fit(X, y)
plot_classifier(X, y, model, proba=True)

# Predict probabilities on training points
prob = model.predict_proba(X)
print("Maximum predicted probability", np.max(prob))


# Set the regularization strength
model = LogisticRegression(C=0.1)

# Fit and plot
model.fit(X, y)
plot_classifier(X, y, model, proba=True)

# Predict probabilities on training points
prob = model.predict_proba(X)
print("Maximum predicted probability", np.max(prob))


# added/edited
X, y = digits.data, digits.target


def show_digit(i, lr=None):
    plt.imshow(
        np.reshape(X[i], (8, 8)), cmap="gray", vmin=0, vmax=16, interpolation=None
    )
    plt.xticks(())
    plt.yticks(())
    if lr is None:
        plt.title("class label = %d" % y[i])
    else:
        pred = lr.predict(X[i][None])
        pred_prob = lr.predict_proba(X[i][None])[0, pred]
        plt.title("label=%d, prediction=%d, proba=%.2f" % (y[i], pred, pred_prob))
        plt.show()


lr = LogisticRegression()
lr.fit(X, y)

# Get predicted probabilities
proba = lr.predict_proba(X)

# Sort the example indices by their maximum probability
proba_inds = np.argsort(np.max(proba, axis=1))

# Show the most confident (least ambiguous) digit
show_digit(proba_inds[-1], lr)

# Show the least confident (most ambiguous) digit
show_digit(proba_inds[0], lr)


# added/edited
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)


# Fit one-vs-rest logistic regression classifier
lr_ovr = LogisticRegression(multi_class="ovr")
lr_ovr.fit(X_train, y_train)

print("OVR training accuracy:", lr_ovr.score(X_train, y_train))
print("OVR test accuracy    :", lr_ovr.score(X_test, y_test))

# Fit softmax classifier
lr_mn = LogisticRegression(multi_class="multinomial")
lr_mn.fit(X_train, y_train)

print("Softmax training accuracy:", lr_mn.score(X_train, y_train))
print("Softmax test accuracy    :", lr_mn.score(X_test, y_test))


# added/edited
toy = pd.read_csv("toy.csv", header=None).to_numpy()
X_train = toy[:, :-1]
y_train = toy[:, -1]
lr_mn = LogisticRegression(C=100, multi_class="multinomial")
lr_mn.fit(X_train, y_train)
lr_ovr = LogisticRegression(C=100, multi_class="ovr")
lr_ovr.fit(X_train, y_train)


# Print training accuracies
print("Softmax     training accuracy:", lr_mn.score(X_train, y_train))
print("One-vs-rest training accuracy:", lr_ovr.score(X_train, y_train))

# Create the binary classifier (class 1 vs. rest)
lr_class_1 = LogisticRegression(C=100)
lr_class_1.fit(X_train, y_train == 1)

# Plot the binary classifier (class 1 vs. rest)
plot_classifier(X_train, y_train == 1, lr_class_1)


# We'll use SVC instead of LinearSVC from now on
from sklearn.svm import SVC

# Create/plot the binary classifier (class 1 vs. rest)
svm_class_1 = SVC()
svm_class_1.fit(X_train, y_train == 1)
plot_classifier(X_train, y_train == 1, svm_class_1)


# added/edited
wine = pd.read_csv("wine.csv", header=None).to_numpy()
X = wine[:, :-1]
y = wine[:, -1]


# Train a linear SVM
svm = SVC(kernel="linear")
svm.fit(X, y)
plot_classifier(X, y, svm, lims=(11, 15, 0, 6))

# Make a new data set keeping only the support vectors
print("Number of original examples", len(X))
print("Number of support vectors", len(svm.support_))
X_small = X[svm.support_]
y_small = y[svm.support_]

# Train a new SVM using only the support vectors
svm_small = SVC(kernel="linear")
svm_small.fit(X_small, y_small)
plot_classifier(X_small, y_small, svm_small, lims=(11, 15, 0, 6))


# added/edited
X, _, y, _ = train_test_split(
    digits.data, digits.target, test_size=0.5, stratify=digits.target
)


# Instantiate an RBF SVM
svm = SVC()

# Instantiate the GridSearchCV object and run the search
parameters = {"gamma": [0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters)
searcher.fit(X, y)

# Report the best parameters
print("Best CV params", searcher.best_params_)


# added/edited
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.5, stratify=digits.target
)


# Instantiate an RBF SVM
svm = SVC()

# Instantiate the GridSearchCV object and run the search
parameters = {"C": [0.1, 1, 10], "gamma": [0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters)
searcher.fit(X_train, y_train)

# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)

# Report the test accuracy using these best parameters
print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))


# added/edited
from sklearn.linear_model import SGDClassifier

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.7, stratify=digits.target
)


# We set random_state=0 for reproducibility
linear_classifier = SGDClassifier(random_state=0)

# Instantiate the GridSearchCV object and run the search
parameters = {
    "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
    "loss": ["hinge", "log_loss"],
}
searcher = GridSearchCV(linear_classifier, parameters, cv=10)
searcher.fit(X_train, y_train)

# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)
print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))
