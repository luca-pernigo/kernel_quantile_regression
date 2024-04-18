# Kernel quantile regression
The kernel_quantile_regression package is an open source implementation of the quantile regressor techique introduced in  [[1]](#1).


Example of kernel quantile regression on the Melbourne temperature data [[2]](#2).
![alt text](https://github.com/luca-pernigo/kernel_quantile_regression/blob/main/plots/melborune_kernel_quantile_regression.png?raw=true)

## Installation
Use the package manager [pip](https://pypi.org/project/kernel-quantile-regression/) to install kernel_quantile_regression.

```bash
pip install kernel-quantile-regression
```

## Usage

```python
from kernel_quantile_regression.kqr import KQR

# create model instance
# specify your quantile q and hyperparameters C and gamma
kqr_1=KQR(alpha=q, C=100, gamma=0.5)

# train model
kqr_1.fit(X_train, y_train)

# predict
kqr_1.predict(X_test)
```



## References
<a id="1">[1]</a> Ichiro Takeuchi, Quoc V. Le, Timothy D. Sears, and Alexander J. Smola. 2006. Non-
parametric Quantile Estimation. Journal of Machine Learning Research 7, 45 (2006),
1231–1264. https://www.jmlr.org/papers/volume7/takeuchi06a/takeuchi06a.pdf

<a id="2">[2]</a> Rob J Hyndman, David M Bashtannyk, and Gary K Grunwald. 1996. Estimating and
visualizing conditional densities. Journal of Computational and Graphical Statistics
5, 4 (1996), 315–336. https://www.jstor.org/stable/1390887

<a id="3">[3]</a> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli, and Rob J.Hyndman. 2016b. Probabilistic energy forecasting: Global Energy Forecasting
Competition 2014 and beyond. International Journal of Forecasting 32, 3 (2016),
896–913. https://www.sciencedirect.com/science/article/abs/pii/S0169207016000133?via%3Dihub

## License
[MIT](https://choosealicense.com/licenses/mit/)