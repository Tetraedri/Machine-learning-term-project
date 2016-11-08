import numpy as np
import matplotlib as mpl
import math

#Data stuff
header = np.genfromtxt("regression_dataset_training.csv", delimiter=',', dtype='str')
header = header[0]

#training data
data = np.genfromtxt("regression_dataset_training.csv", delimiter=',', dtype=int)
data = np.delete(data, (0), axis=0)
indexes = data[:, 0]
data = np.delete(data, (0), axis=1)
ratings = data[:,len(header)-2]
data = np.delete(data, (len(data[0])-1), axis=1)

#test data
test_data = np.genfromtxt("regression_dataset_testing.csv", delimiter=',', dtype=int)
test_data = np.delete(test_data, (0), axis=0)
test_indexes = test_data[:, 0]
test_data = np.delete(test_data, (0), axis=1)

def square(x): return x*x

def closest_k(base, unknown, k):
    a = base - unknown
#    print('erotus', a)
    a = np.array(list(map(square, a)))
#    print('squaret', a)
    a = np.sum(a, axis=1, dtype=np.int32)
#    print('summat', a)
    a = np.array(list(map(math.sqrt, a)))
    a = np.array(list(enumerate(a)), dtype=[('x', int),('y', float)])
#    print('indeksoitu',a)
    a.sort(axis=0, order='y')
#    print('sortattu',a)
    k = min(k, len(base))
    return a[:k] #a[:,0]]

def model(train, test):
    results = []
    index = test_indexes[0]

    for i in range(0, len(test[:,0])):
        close = closest_k(train, test[i], 1)
        prediction = 0.0
        weight = 0.0
        scaler = 0.0

        for pair in close:
            scaler = max(scaler, pair[1])

        for pair in close:
            train_rating = ratings[pair[0]]
            dist = ratings[pair[1]]

            # abs(distance - max-dist) weight
            weight_adjusted = abs(dist - scaler)
            prediction += train_rating*weight_adjusted
            weight += weight_adjusted

            # 1/distance weight
            #dist_adjusted = max(dist, 0.01)
            #weight += 1/dist_adjusted
            #prediction += train_rating*(1/dist_adjusted)

        if weight != 0:
            prediction = prediction/weight

        #Rounding optional
        rounded_prediction = prediction #int(round(prediction))

        results += [[str(index), str(rounded_prediction)]]
        index = index + 1

    np.savetxt('reg_result.csv', results, delimiter=',', fmt='%s')
    return results[:20]

print(model(data, test_data))




#a = np.array([[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]])
#b = np.array([2, 2, 2, 2, 2, 2])

#z = closest_k(a, b, 4)
