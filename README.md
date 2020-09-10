A repository containing work with support vector machines (SVMs) and least squares support vector machines (LS-SVMs) in particular. Included are experiments with classification, function estimation, (automated) parameter tuning, Bayesian optimisation, automatic relevance detection, robust regression, time series prediction, large-scale problems (using the Nyström method or network committees) and multiclass classification. Spectral clustering and kernel principal component analysis are considered as well. Some basic tests on well-known datasets (such as the Pima indian diabetes -, California housing - and Santa Fe chaotic laser data) are applied.

---

## Classification

### Two Gaussians

For a binary classifier where the distributions are (assumed or known to be) Gaussian with equal covariance matrices the decision boundary that maximises the posterior prob- ability P(Ci|x) becomes linear. This is independent of the amount of overlap. Trying to get a better boundary would lead to overfitting. In this particular example where Σxx = I one ends up with a perpendicular bisector of the segment connecting the two cluster means (−1,−1) and (1, 1), which gives f (x) = −x as a decision boundary.

<p align="center">
<img src="https://github.com/BrunoVDK/support-vector-machines/blob/master/report/res/readme1.png?raw=true">
</p>

### Support Vector Machine Classifier

To deal with the non-linearly separable classification prob- lem in the example one solves the following minimisation problem, where the hyperparameter C controls the trade- off between maximising the margin and making sure that the data lies on the correct side of that margin :

<p align="center">
<img src="https://github.com/BrunoVDK/support-vector-machines/blob/master/report/res/readme2.png?raw=true">
</p>

A dual form can be expressed by making use of Lagrange multipliers which in this context are also called *support values*. Any data points for which the corresponding support value isn't zero are called *support vectors*. These lie close to the boundary or are misclassified.

For the given toy dataset it can readily be seen in figure 2 (top row) that for a decreasing value of C the margin does become larger and that less ‘slacking’ is allowed for. By adding data points close to the boundary or at the ‘wrong’ side of it the margin usually changes a lot and the new data points nearly always become support vectors.
The σ2 hyperparameter in the case of an RBF kernel controls the locality of each data point. As it increases
the feature space mapping becomes smoother, producing simpler decision boundaries (figure 2, third row). This
relates to the bias-variance tradeoff. For σ2 small enough any two classes get separated in a very strict way - the
Gaussian kernels are universal. In the limit the kernel matrix becomes an identity matrix. For large bandwidths
the reach of each data point becomes large and the decision boundary can become near-linear as the density areas of each class end up competing. In the limit the kernel matrix is filled with ones and the classification fails (in the demo this can happen at σ2 = 105, in the LS-SVM toolbox a much larger value is needed). As for C, it works the same as before, prioritising either larger margins or lower misclassification rates. When σ2 is large a bigger value of C can make the model more complex again.

<p align="center">
<img src="https://github.com/BrunoVDK/support-vector-machines/blob/master/report/res/readme15.png?raw=true">
</p>

The RBF kernel approach can generate models having misclassification rates that are lower than the classic linear kernel approach does as decision boundaries that are nonlinear in the input space can be learned. It also tends to use more data points as support vectors. This makes the model less compact (computationally efficient) and if the data is linearly separable it is unnecessary. Deciding whether the model generalises better can be done through evaluation on a test set.

### Least-squares support vector machine classifier

Figure 3 depicts results on the iris dataset using various polynomial kernels with t = 1. This t is called the constant and it controls the importance of higher - versus lower-order terms in the polynomial as can be deduced from the application of the binomial theorem to the definition of the kernel :

<p align="center">
<img src="https://github.com/BrunoVDK/support-vector-machines/blob/master/report/res/readme3.png?raw=true">
</p>

The effect of t is not very noticeable for small values of d while for the higher degrees it tends to make the
decision boundary more complex.

When the degree δ is 1 the feature map is linear which is equivalent to the classic non-linear SVM problem while for higher degrees a boundary of increasing complexity is learned such that for δ ∈ {3,4} no data points are misclassified. To make it less likely that the model overfits a lower δ is likely to be preferable (this corresponds to the application of Occam’s razor).

In the case of RBF kernels one can see in figure 4 that for a γ value of 1 any σ2 between 0.1 and 1 performs well. The same interval works for γ when σ2 = 1. This corresponds to the results of the provided sample script where the base used in the semilog plot happens to be the natural logarithm (instead of using base 10). For large σ values (say, 25) one class may overtake the other one. The experiment is of limited value as only few parameter combinations were tried.

<p align="center">
<img src="https://github.com/BrunoVDK/support-vector-machines/blob/master/report/res/readme4.png?raw=true">
</p>

To properly find good parameter values a more systematic approach is in order. The idea is to search through the parameter space and evaluate the results on a validation set rather than on the test set (which should be used for nothing but the evaluation of the finished model). A validation set can be constructed in a few ways the results of which are illustrated in figure 6, where error estimates in the parameter space are shown. Random splitting i.e. splitting the input data randomly in a training - and a validation set is a way that isn’t particularly robust. An improvement upon it is k-fold validation where the input data is randomly split into k folds which are considered as a validation set in turn such that all the input data can be used for parameter tuning. When k = N with N the number of input samples this is called leave-one-out cross-validation.

To some extent deciding what k value to use parallels pre- vious discussions about other parameters since there’s a bias-variance tradeoff ; while larger k values should pro- vide a better estimate of the error they also suffer from higher variance as the result depends more on how rep- resentative the input data is. This becomes more impor- tant when the number of input data is small. Finally, k shouldn’t be too small (such that the training sets re- main large enough), it should preferably be a divisor of N (though this is not of primary importance) and if compu- tational expense is an issue it cannot be large as many models would have to be generated (which admittedly can be addressed).

<p align="center">
<img src="https://github.com/BrunoVDK/support-vector-machines/blob/master/report/res/readme5.png?raw=true">
</p>

Automated tuning uses these validation methods in conjunction with a search procedure to find useful combinations of parameters. This search strategy has to deal with a non-convex problem and can be a simple grid
search (performing validation for every combination of parameters specified by a grid partitioning the parameter space) or the Nelder-Mead method. The latter aims to find the minimum of a function without needing its derivative by considering a simplex which is updated iteratively until it wraps around the minimum. It tends to execute faster than grid search especially when the number of parameters is high, though to address this one could also consider a variant of grid search that starts with a grid of limited granularity until it finds a promising region in the parameter space on which grid search is performed again.

In the LS-SVM toolbox a technique called coupled simulated annealing is used. This is a global optimisation technique which couples together several simulated annealing processes (inspired by coupled local minimisers’ effectiveness compared to multi-start gradient descent optimisation). Its output is subsequently provided to either a grid search - or a Nelder-Mead algorithm. Results are the following :

<p align="center">
<img src="https://github.com/BrunoVDK/support-vector-machines/blob/master/report/res/readme6.png?raw=true">
</p>

Simulated annealing is a stochastic process such that the parameters end up varying quite a bit. The costs do not since they are minimised. A histogram of the results gives a more complete picture and shows that there are a few outliers that make the average γ and σ2 large :

<p align="center">
<img src="https://github.com/BrunoVDK/support-vector-machines/blob/master/report/res/readme7.png?raw=true">
</p>

Results of the automated tuning process correspond to the contour plots given in figure 6. As for the runtimes,
the grid search ended up being twice as slow as the simplex method.

<p align="center">
<img src="https://github.com/BrunoVDK/support-vector-machines/blob/master/report/res/readme8.png?raw=true">
</p>

An ROC curve can be generated for any model. One calculates the result of the model applied on every test data point and uses the results as thresholds for which the true positive rate (TPR) is plotted in function of the false positive rate (FPR). The area under the curve (AUC) can then be used to gauge the effectiveness of the classifier. For γ = 0.037622,σ2 = 0.559597 it happens to equal 1 indicating a perfect classifier. If the sign function had been used to classify test data there would be a misclassification error (the FPR would be 0.1).

<p align="center">
<img src="https://github.com/BrunoVDK/support-vector-machines/blob/master/report/res/readme9.png?raw=true">
</p>

Using a Bayesian approach (considered in greater detail in the second part of this report) which requires a prior denoting the probability of occurrence of each class it is possible (by applying Bayes’ theorem) to estimate the probability that a given data point belongs to the posi- tive or the negative class. A corresponding plot of these posterior probabilities for the same model used to cal- culate the ROC curve is shown in figure 7. The prior is taken to be 0.67 (the classes are unbalanced).

### Ripley dataset

<p align="center">
<img src="https://github.com/BrunoVDK/support-vector-machines/blob/master/report/res/readme10.png?raw=true">
</p>

The Ripley dataset is a 2-dimensional so it can easily be visualised (figure 8). The data for each class is generated by a mixture of two Gaussian distributions and it is not linearly separable in the input space. The distributions were chosen as to ‘allow a best-possible error rate of about 8%’, each component has the same covariance matrix (basically a 2-component version of the introductory example). The data is a ‘realistic’ toy problem and it is rather noisy.

An LS-SVM with a linear kernel makes for a classification rate of around 10% while an RBF kernel performs a tad better at a rate of about 9% (and so does a polynomial one3). The ROC curves give a more complete picture and aren’t shown here, but the AUC was the highest for the RBF kernel (about 0.97) with accuracy being the highest at a threshold a bit below zero. In practice (when met with real-life datasets) operating points on the ROC curve calculated on the validation set may be chosen depending on the cost of each type of misclassification.

Steve Gunn’s MATLAB toolbox which is also used for regression in the second part of the report provides functions for generating classical SVMs and some more kernels that can be experimented with though it provides no automated tuning method for these. The results for linear -, RBF and polynomial kernels are about the same though there are less support vectors. Results after manual tuning (through the uiclass interface) are shown in figure 9.

<p align="center">
<img src="https://github.com/BrunoVDK/support-vector-machines/blob/master/report/res/readme11.png?raw=true">
</p>

Since the dataset is Gaussian an RBF kernel should be an appropriate pick. As it’s also a noisy dataset a smoother decision surface (higher σ2 and/or lower γ) should be opted for. Of course in this case the optimal classifier (in terms of maximisation of the posterior probabilities) can be defined based on Bayes’ theorem.

### Wisconsin breast cancer dataset

With 30 features for each of the observations the Wisconsin breast cancer diagnosis dataset isn’t trivial to visualise. MATLAB has an implementation of t-SNE or two principal components determined through PCA could be plotted (see figure 11). Distributions of some of the features are shown below. The two classes denote benign - or malignant cancer tissue. In contrast with the previous dataset this one’s a bit unbalanced (62.5% belongs to the negative class) which can be accounted for through bias correction4 or by using weights in the SVM formulation to give more emphasis to the terms associated to the minority class.

<p align="center">
<img src="https://github.com/BrunoVDK/support-vector-machines/blob/master/report/res/readme12.png?raw=true">
</p>

A simple automated tuning experiment of an LS-SVM with linear -, polynomial - and RBF kernels yielded average AUCs of 0.996 ± 0.001, 0.937 ± 0.058 and 0.995 ± 0.002 respectively. It may be surprising that a linear kernel outperformed a non-linear one - the number of features is larger and there’s not that many data points so a simpler decision boundary may help to avoid overfitting since Cover’s theorem implies that it is more likely that the data is linearly separable when the number of dimensions is high (and the number of data points is low).

<p align="center">
<img src="https://github.com/BrunoVDK/support-vector-machines/blob/master/report/res/readme13.png?raw=true">
</p>

As some of the features seem less informative than others it seemed worthwhile to remove some of them either through PCA or a more sophisticated approach like automatic relevance determination. A basic experiment ended up with the latter selecting 21 features. Training the model on these resulted in an AUC of 0.9973. The results with a larger number of tuning and training runs as well as a comparison with a basic PCA approach are shown in table 2.

<p align="center">
<img src="https://github.com/BrunoVDK/support-vector-machines/blob/master/report/res/readme14.png?raw=true">
</p>

### Diabetes dataset

A 2-dimensional visualisation of the 8-dimensional Pima indians dataset (figure 11) reveals it to be more difficult
than the previous one (distributions of the input variables tell the same story). Mean error rates of 21.7% ± 1.4, 27.6% ± 5.3 and 22.2% ± 1.8 were obtained with a linear, polynomial and RBF kernel (averaged over 10 runs 5
using tuned parameters) . The average AUCs were 0.844 ± 0.001, 0.721 ± 0.112 and 0.848 ± 0.004. An RBF kernel appeared most performant. Automatic relevance determination did not prune any of the input variables which made subsequent poor results with PCA not surprising.

The original paper dealing with this dataset used a custom algorithm related to neural networks. They used all training examples and reached an accuracy of 76%. The performance of an LS-SVM with an RBF kernel trained on the full training set (named total) was typically comparable.

A quick comparison was done with k-Nearest Neighbour (kNN) and a random forest classifier. For k ∈ [1, 10] the nearest neighbour algorithm had an accuracy below 70% which is significantly worse while the random forest classifier (TreeBagger in MATLAB) did a bit better with an accuracy of about 73%.

It looks like the accuracy is amongst the better ones. Fisher Discriminant Analysis (FDA) appears to perform
alright as well but isn’t experimented with here since it seemed redundant due to the close relation with LS-
SVMs. Maybe more advanced neural networks might do better. A simple neural network constructed in MATLAB 6
also got into the 70% accuracy .
