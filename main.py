import numpy as np

# functions
from scipy.optimize import linprog, OptimizeResult


def generate_tab_initial(a, b, c):
    z = np.array([0])
    c = np.append(c, z, axis=0)
    nl, nc = a.shape
    for i in range(0, nl):
        c = np.append(c, [0], axis=0)
    diag = np.eye(nl, nl)
    a = np.append(a, diag, axis=1)
    a = np.append(a, b.reshape((len(b), 1)), axis=1)
    a = np.append(a, c.reshape(1, len(c)), axis=0)
    return a


def positive(v):
    test = False
    min = 0
    index = 0
    for i in range(0, len(v)):
        if v[i] < min:
            test = True
            index = i
            min = v[i]
    return test, min, index


def rapport_min(a, b):
    min = abs(max(b))  # place holder solution
    index = -1
    for i in range(0, len(a)):
        if a[i] > 0 and b[i] > 0:
            if b[i] / a[i] < min:
                min = b[i] / a[i]
                index = i
    if index == -1:
        return False, index
    else:
        return True, index


def pivot_gauss(a, l, c):
    nl, nc = a.shape
    a[l, :] = a[l, :] / a[l][c]
    for i in range(0, nl):
        if i != l:
            coficient = a[i][c] if a[i][c] != 0 else 0
            for j in range(0, nc):
                a[i][j] = a[i][j] - coficient * a[l][j]
    return a


def simplex(c, a, b, maxiter=100):
    status = 0
    messages = {0: "Optimization terminated successfully.",
                1: "Iteration limit reached.",
                2: "Optimization failed. Unable to find a feasible"
                   " starting point.",
                3: "Optimization failed. The problem appears to be unbounded.",
                4: "Optimization failed. Singular matrix encountered."}
    nit = 0
    complete = False
    tab = generate_tab_initial(a, b, c)
    nl, nc = tab.shape
    while not complete:
        colone_pivot_existe, _, colone_pivot = positive(tab[nl - 1, :])
        if not colone_pivot_existe:
            status = 0
            complete = True
        ligne_pivot_existe, ligne_pivot = rapport_min(tab[:, colone_pivot], tab[:, nc - 1])
        if not ligne_pivot_existe:
            status = 3
            complete = True
        if nit >= maxiter:
            # Iteration limit exceeded
            status = 1
            complete = True
        else:
            tab = pivot_gauss(tab, ligne_pivot, colone_pivot)
            nit += 1

    sol = {
        'x': tab[:, nc - 1],
        'fun': -tab[nl - 1, nc - 1],
        # 'slack': slack,
        # 'con': con,
        'status': status,
        'message': messages[status],
        'nit': nit,
        'success': status == 0}

    return OptimizeResult(sol)


# main


a = np.array([[1, 1, 2], [2, 0, 3], [2, 1, 3]])
b = np.array([4, 5, 7])
c = np.array([-3, -2, -4])
print(simplex(c, a, b))
print("===============================================")
print(linprog(c, a, b))
