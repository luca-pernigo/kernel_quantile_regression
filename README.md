# Kernel quantile regression
The kernel_quantile_regression package is an open source implementation of the quantile regressor techique from https://jmlr.org/papers/volume7/takeuchi06a/takeuchi06a.pdf

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


## License
[MIT](https://choosealicense.com/licenses/mit/)