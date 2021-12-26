import numpy as np

# functions
from scipy.optimize import linprog, OptimizeResult


def generate_tab_initial(a, b, c):  # Combine les 3 tableau en une seul matrice
    z = np.array([0])  # Z est initaliser a 0
    c = np.append(c, z, axis=0)  # Z est ajouter a la fin du vecteur
    nl, nc = a.shape
    for i in range(0, nl):  # ajout des variable intermediare
        c = np.append(c, [0], axis=0)
    diag = np.eye(nl, nl)  # creation de la matrice diagonale (identité) pour la matrice intermidaire
    a = np.append(a, diag, axis=1)  # concataination entre a et diag sur l'axe 1
    a = np.append(a, b.reshape((len(b), 1)), axis=1)  # concataination entre a et b sur l'axe 1
    a = np.append(a, c.reshape(1, len(c)), axis=0)  # concataination entre a et c sur l'axe 0
    return a


def positive(v):
    test = False  # initialisation des variables
    min = 0
    index = 0
    for i in range(0, len(v)):  # on parcoure la derniere ligne de la matrice
        if v[i] < min:  # check if v[i] is smaler than min to find the smalest negative number
            test = True  # test is switched to true
            index = i  # index takes i
            min = v[i]  # min est ecraser avec v[i]

    return test, min, index  # on retourne test,min,index


def rapport_min(a, b):
    min = abs(max(b))  # min est initialiser a la valeur a absolue du max du vecteur b
    index = -1  # index est intialiser a -1
    for i in range(0, len(a)):
        if a[i] > 0 and b[i] > 0:  # on verfie que ni a[i] ni b[i] sont negative
            if b[i] / a[i] < min:
                min = b[i] / a[i]  # min est ecraser
                index = i  # index prend la valeur de i

    return index != -1, index  # index == -1 retour False se aucune ligne pivaut n'est trouver


def pivot_gauss(a, l, c):
    nl, nc = a.shape
    a[l, :] = a[l, :] / a[l][c]  # on normalise la ligne pivot (a[l][c] = 1)
    for i in range(0, nl):
        if i != l:  # on soute la ligne pivot
            coficient = a[i][c] if a[i][c] != 0 else 0  # coficient = a[i][c] si  a[i][c] != 0
            for j in range(0, nc):
                a[i][j] = a[i][j] - coficient * a[l][j]  # ligne_i - coficient*ligne_pivot
    return a


def simplex(c, a, b, maxiter=100):
    status = 0  # status et initaliser a zero
    messages = {0: "Optimization terminated successfully.",
                1: "Iteration limit reached.",
                3: "Optimization failed. The problem appears to be unbounded."}
    # messages est un dictionaire de message d'exectution baser sur la documentation de linprog
    nit = 0  # le nombre d'iteration
    complete = False  # statue de completion
    vdnl, snc = a.shape  # vdnl represente le nombre de variable de desision snc represente le nombre de variable intermediare
    variables = np.full(vdnl + snc, -1.0)
    # creation d'un vecteur variables de taille vndl+snc intialiser a -1 ce vecteur vas nous permetre suivre l'entre et la sortie de variables de la base
    for i in range(vdnl, vdnl + snc):
        variables[i] = i - vdnl  # entrer des variable intermediare dans la base
    tab = generate_tab_initial(a, b, c)  # generation de notre matrice initial
    nl, nc = tab.shape  # on trouve le nombre de ligne et de colone
    while not complete:
        if nit >= maxiter:  # Si le nombre d'iteration max et ateint on considere que l'operation est complete
            # Iteration limit exceeded
            status = 1
            complete = True

        else:
            colone_pivot_existe, _, colone_pivot = positive(tab[nl - 1, :])  # on recherce la colone pivot
            if not colone_pivot_existe:  # Si la colone pivot n'existe pas le status est mis a 0 et complete est mis a True
                status = 0
                complete = True

            ligne_pivot_existe, ligne_pivot = rapport_min(tab[:, colone_pivot], tab[:, nc - 1])
            if not ligne_pivot_existe:      # Si la ligne pivot n'existe pas le status est mis a 3 et complete est mis a True
                status = 3
                complete = True

            for i in range(vdnl + snc):                 #on parcour le vecteur variable
                if variables[i] == ligne_pivot:
                    variables[i] = -1                   #la variable[i] sort de la base

            variables[colone_pivot] = ligne_pivot       #La variables[colone_pivot] entre en base
            tab = pivot_gauss(tab, ligne_pivot, colone_pivot)   #pivot de gauss
            nit += 1             #nombre d'iteration + 1

    for i in range(vdnl + snc):              #on parcour le vecteur variable
        if variables[i] == -1:               #la variable[i] est  -1 on la remplace avec 0
            variables[i] = 0
        else:
            variables[i] = tab[int(variables[i]), nc - 1]       #les variable de  base prene leur valeur qui se trouve dans la derniere colone de tab

    sol = {
        'x': variables[:vdnl],                  #x represente les variable xi que on avais dans le debut
        'fun': -tab[nl - 1, nc - 1],            #fun est -z
        'slack': variables[vdnl:],              #stack sount les variable intermediaire
        'status': status,                       #le satue de complition de probleme
        'message': messages[status],            #Le message baser selon le statue
        'nit': nit,                             #nombre d'iteration
        'success': status == 0}                 #retourne True si status est egale a 0

    return OptimizeResult(sol)                  #permet d'avoir le meme format que linprog


# main

if __name__ == '__main__':
    a = np.array([[1, 1, 2], [2, 0, 3], [2, 1, 3]])
    b = np.array([4, 5, 7])
    c = np.array([-3, -2, -4])
    print(simplex(c, a, b))
    print("===============================================")
    print(linprog(c, a, b))
