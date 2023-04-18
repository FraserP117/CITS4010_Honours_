import numpy as np
import matplotlib.pyplot as plt

def logistic(x, L, k, x_m):
	''' 
	x:   the domain value
	L:   the supremmum
	k:   growth rate
	x_m: x-value of the function's midpoint
	'''
	return L / (1 + np.exp(- k * (x - x_m)))

if __name__ == "__main__":

	X = np.linspace(0, 40, 1000)
	Y_1 = []
	Y_2 = []
	Y_3 = []
	
	for i in X:
		'''
		y_1 = logistic(i, 1, 1, 0)
		Y_1.append(y_1)
		
		y_2 = logistic(i, 1, 2, 0)
		Y_2.append(y_2)
		'''

		y_3 = - logistic(i, 1, 1/12, 10) + 1
		Y_3.append(y_3)

	# plt.plot(X, Y_1, color = "blue")
	# plt.plot(X, Y_2, color = "red")
	plt.plot(X, Y_3, color = "green")
	plt.show()
