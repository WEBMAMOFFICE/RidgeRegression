import matplotlib.pyplot as plt
from sklearn import linear_model
y = [1, 2, 3, 4, 5]
X = [[1.1, 2.1, 3.1, 4.1, 5.1], [1.2, 2.2, 3.2, 4.2, 5.2], [1.3, 2.3, 3.3, 4.3, 5.3],
     [1.4, 2.4, 3.4, 4.4, 5.4], [1.5, 2.5, 3.5, 4.5, 5.5]]
lr = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)
lr.fit(X, y)
lrR2 = lr.score(X, y)
Y0 = [lr.coef_[0] * 0 + lr.intercept_, 6]
Y6 = [lr.coef_[0] * 0 + lr.intercept_, lr.coef_[0] * 6 + lr.intercept_]
print("coef_ value = ", lr.coef_)
print("intercept_ value = ", lr.intercept_)
print("Coeficient of Determination R² = ", round(lrR2, 2))
# Plot outputs
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter([1, 1, 1, 1, 1], X[0],  color='red', linewidth=3)
ax.scatter([2, 2, 2, 2, 2], X[1],  color='red', linewidth=3)
ax.scatter([3, 3, 3, 3, 3], X[2],  color='red', linewidth=3)
ax.scatter([4, 4, 4, 4, 4], X[3],  color='red', linewidth=3)
ax.scatter([5, 5, 5, 5, 5], X[4],  color='red', linewidth=3)
ax.scatter(6, lr.coef_[0] * 6 + lr.intercept_,  color='#FF00FF', linewidth=3)
ax.plot(Y0, Y6, color='green', linewidth=3)
ax.set_ylim(-1, 8)
ax.set_xlim(-1, 8)
plt.show()
