import matplotlib.pyplot as plt
from sklearn import linear_model

X1 = [1, 2.5, 2, 4, 5]
X2 = [1.5, 3, 2.5, 4.5, 5]
X3 = [2, 1, 3.5, 4, 5]
X4 = [1, 3, 2, 5, 4]
X5 = [1.5, 2.5, 4, 3, 5.5]

y = [1, 2, 3, 4, 5]
X = [[X1[0], X2[0], X3[0], X4[0], X5[0]], [X1[1], X2[1], X3[1], X4[1], X5[1]],
     [X1[2], X2[2], X3[2], X4[2], X5[2]], [X1[3], X2[3], X3[3], X4[3], X5[3]],
     [X1[4], X2[4], X3[4], X4[4], X5[4]]]
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
    n1 = int("FF0000", 16) * ((n + 2) * 2)
    color = "#%s" % (hex(n1)[2:],)
    ax.scatter(y, [X[0][n], X[1][n], X[2][n], X[3][n], X[4][n]],  color=color[0:7], linewidth=3)
    ax.plot(y, [X[0][n], X[1][n], X[2][n], X[3][n], X[4][n]], ls='solid', lw=1.4, aa=True, color=color[0:7])
    ax.scatter(6, sum(lr.coef_) * 6 + lr.intercept_,  color='#FF00FF', linewidth=3)
Y0 = [sum(lr.coef_) * 0 + lr.intercept_, 6]
Y6 = [sum(lr.coef_) * 0 + lr.intercept_, sum(lr.coef_) * 6 + lr.intercept_]
ax.plot(Y0, Y6, color='green', linewidth=3)
ax.set_ylim(-1, 8)
ax.set_xlim(-1, 8)
plt.show()
