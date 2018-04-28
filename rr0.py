import matplotlib.pyplot as plt
from sklearn import linear_model
y = [1, 2, 3, 4, 5]
X = [[1.1, 1.3, 1.7, 1.4, 1.9], [2.1, 2.3, 2.7, 2.4, 2.9], [3.1, 3.3, 3.7, 3.4, 3.9],
     [4.1, 4.3, 4.7, 4.4, 4.9], [5.1, 5.3, 5.7, 5.4, 5.9]]
lr = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)
lr.fit(X, y)
lrR2 = lr.score(X, y)
print("coef_ value = ", lr.coef_)
print("intercept_ value = ", lr.intercept_)
print("Coeficient of Determination R² = ", round(lrR2, 2))
# Plot outputs
fig = plt.figure()
ax = fig.add_subplot(111)
for n in range(0, len(X)):
    ax.scatter(y, [X[0][n], X[1][n], X[2][n], X[3][n], X[4][n]],  color='red', linewidth=3)
    ax.scatter(6, sum(lr.coef_) * 6 + lr.intercept_,  color='#FF00FF', linewidth=3)
Y0 = [sum(lr.coef_) * 0 + lr.intercept_, 6]
Y6 = [sum(lr.coef_) * 0 + lr.intercept_, sum(lr.coef_) * 6 + lr.intercept_]
ax.plot(Y0, Y6, color='green', linewidth=3)
ax.set_ylim(-1, 8)
ax.set_xlim(-1, 8)
plt.show()
