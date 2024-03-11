# Devoir 1
import matplotlib.pyplot as plt
import numpy as np

gamma = [0.0137, 0.0274, 0.0434, 0.0866, 0.137, 0.274, 0.434, 0.866, 1.37, 2.74, 4.34, 5.46, 6.88]
etha = [3220, 2190, 1640, 1050, 766, 490, 348, 223, 163, 104, 76.7, 68.1, 58.2]

def f(etha0, betha, lambda_, gamma_i):
    res = 1 + (lambda_**2) * (gamma_i **2)
    res = res**((betha - 1)/2)
    return etha0 * res

def epsilon(i, etha0, lambda_, betha):
    return abs(f(etha0, betha, lambda_, gamma[i]) - etha[i])

def g_hat(etha0, lambda_, betha):
    abs_err = 0
    for i in range(len(gamma)):
        abs_err += epsilon(i, etha0, lambda_, betha)
    return abs_err

def g_(etha0, lambda_, betha):
    squ_err = 0
    for i in range(len(gamma)):
        squ_err += epsilon(i, etha0, lambda_, betha)**2
    return squ_err



def hypercube_latin(n_eval, dim=3, borne_inf=0, borne_sup=20):
    # Génération des points via un hypercube latin
    intervalle = np.linspace(borne_inf, borne_sup, n_eval+1)
    grille = np.empty((n_eval, dim))
    for d in range(dim):
        pas = np.random.permutation(n_eval)
        for i in range(n_eval):
            grille[i, d] = np.random.uniform(intervalle[pas[i]], intervalle[pas[i]+1])
    return grille

def trouver_meilleur_point(f, grille):
    meilleur_score = np.inf
    meilleur_point = None
    for i in range(len(grille[:,0])):
        t = [grille[i,0],grille[i,1], grille[i,2]]
        evaluation = f(520*t[0],14*t[1],0.038*t[2])
        if evaluation < meilleur_score:
            meilleur_score = evaluation
            meilleur_point = t
    return np.array(meilleur_point), meilleur_score

########################################################################

def Algo2(f, x0, budget, D, f_0):
    ameliorations = [[int(budget/50), f_0]]
    deltak = 1
    xk     = x0
    f_xk   = f_0
    tau    = 1/2
    k      = int(budget/50)
    while k < budget:
        moving = False
        for i in range(len(D)):
            d = D[i]
            t = xk + deltak * d
            f_eval = f(520*t[0],14*t[1],0.038*t[2])
            k += 1
            if f_xk > f_eval:
                ameliorations.append([k, f_xk])
                xk = t
                f_xk = f_eval
                deltak *= (1/tau)
                moving = True
                D = np.concatenate((np.array([D[i]]), D[:i], D[i+1:]))
                break
            if k == budget:
                break
        if not moving:
            deltak *= tau
    return xk, f_xk, ameliorations

D1 = [[1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [-1, 0, 0],
      [0, -1, 0],
      [0, 0, -1]]
D1 = np.array(D1)
print(D1)

########################################################################

def generer_vecteur_normalise(dim=3):
    # Générer un vecteur aléatoire
    vecteur = np.random.randn(dim)
    # S'assurer que le vecteur n'est pas nul
    while np.all(vecteur == 0):
        vecteur = np.random.randn(dim)
    # Normaliser le vecteur
    norme = np.linalg.norm(vecteur)
    vecteur_normalise = vecteur / norme
    return vecteur_normalise

def Algo3(f, x0, budget, In, f_0):
    ameliorations = [[int(budget/50), f_0]]
    deltak = 1
    xk     = x0
    f_xk   = f_0
    tau    = 1/2
    k      = int(budget/50)
    B = np.zeros((3, 3))
    while k < budget:
        moving = False
        v = generer_vecteur_normalise(3)
        H = In - 2 * np.outer(v, v.T)
        for j in range(len(H[0,:])):
            norme_Hj = np.linalg.norm(H[:,j], ord=np.inf)
            B[:,j] = deltak * np.round(H[:,j]/(deltak * norme_Hj))
        D = np.transpose(np.concatenate([B, -B], axis=1))
        for i in range(len(D)):
            d = D[i]
            t = xk + deltak * d
            f_eval = f(520*t[0],14*t[1],0.038*t[2])
            k += 1
            if f_xk > f_eval:
                ameliorations.append([k, f_xk])
                xk = t
                f_xk = f_eval
                deltak *= (1/tau)
                moving = True
                D = np.concatenate((np.array([D[i]]), D[:i], D[i+1:]))
                break
            if k == budget:
                break
        if not moving:
            deltak *= tau
    return xk, f_xk, ameliorations

I3 = [[1, 0, 0],
      [0, 1, 0],
      [0, 0, 1]]

########################################################################
n_evals = [50, 100, 250, 500, 750 ,1000, 2500, 5000, 7500, 10000, 20000, 50000]
n_starting_points = 8
for n_eval in n_evals:
    grille = hypercube_latin(n_eval, dim=3, borne_inf=0, borne_sup=20)
    best_x_1, best_f_1 = trouver_meilleur_point(g_hat, grille)
    grille_partielle = hypercube_latin(int(n_eval/50), dim=3, borne_inf=0, borne_sup=20)
    starting_point, f_starting_point = trouver_meilleur_point(g_hat, grille_partielle)

    best_x_2, best_f_2, log_2 = Algo2(g_hat, starting_point, n_eval - int(n_eval/50), D1, f_starting_point)
    best_x_3, best_f_3, log_3 = Algo3(g_hat, starting_point, n_eval - int(n_eval/50), I3, f_starting_point)
    print(f"Algo 1 : Pour x0 = [{starting_point[0]:.2f}, {starting_point[1]:.2f}, {starting_point[2]:.2f}] et budget = {n_eval} on a fbest = {best_f_1:.5f}")
    print(f"Algo 2 : Pour x0 = [{starting_point[0]:.2f}, {starting_point[1]:.2f}, {starting_point[2]:.2f}] et budget = {n_eval} on a fbest = {best_f_2:.5f}")
    print(f"Algo 3 : Pour x0 = [{starting_point[0]:.2f}, {starting_point[1]:.2f}, {starting_point[2]:.2f}] et budget = {n_eval} on a fbest = {best_f_3:.5f}")

########################################################################
