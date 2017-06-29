from scipy.sparse import lil_matrix
num_i = 20
e = lil_matrix((num_i, num_i))
print(e[0].toarray()[0])