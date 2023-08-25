#Numerical calculations modules
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit, bisect
from scipy.special import factorial, factorial2

#Plotting module
import matplotlib.pyplot as plt

#Quality of life modules
import os
import time
from tqdm import tqdm

#Symbolic calculations modules
# from sympy import Matrix, Symbol, pprint

#Dimensional parameters
eV = 1.602176634 * 10**(-19)
keV = eV*10**3
MeV = eV*10**6
GeV = eV*10**9
TeV = eV*10**12
h_bar = 6.582119569*10**(-16) * eV
c = 299792458
m_chi = 1*TeV
m_phi = 1*GeV
delta = 100*keV
alpha = 1/137
v = 130
r_0 = 1/(alpha*m_chi*c/h_bar)

l = 0

#Dimensionless parameters
# eps_v = v/c/alpha
# eps_v = 0.15
# eps_delta = np.sqrt(2*delta/m_chi)/alpha
# eps_delta = 0.1
# eps_phi = m_phi/m_chi/alpha
# eps_phi = 0.2

#Yukawa potential
def Yukawa(r, eps_phi):
    return np.exp(-eps_phi*r)/r

def solution(eps_v, eps_delta, eps_phi, V = Yukawa, print_out=False, plot_sol=False, plot_fit=False, plot_chi_sq=False, numerical_accuracy = 5):

    damping = False
    below_threshold = (eps_v <= eps_delta)
    if print_out:
        print("Below threshold: ", below_threshold)

    r_min = 0.001
    r_max = 1000
    # if below_threshold:
    #     r_max /= 100
    N = 100000
    r_tab = np.geomspace(r_min, r_max, N)
    if below_threshold:
        eps_damping = np.min((eps_v, np.sqrt(eps_delta**2 - eps_v**2)))/100

    #Schrodinger equation
    def fun(r, X):
        # print(r, X)
        ### DAMPING
        if damping:
            equation = np.array([
                                [0, 0, 1, 0],
                                [0, 0, 0, np.exp(-eps_damping*r**2)*1],
                                [l*(l+1)/r**2 - eps_v**2, -V(r, eps_phi), 0, 0],
                                [np.exp(-eps_damping*r**2)*(-V(r, eps_phi)), np.exp(-eps_damping*r**2)*(l*(l+1)/r**2 + eps_delta**2 - eps_v**2), 0, 0]
                                ])
        ### NO DAMPING
        else:
            equation = np.array([
                                [0, 0, 1, 0],
                                [0, 0, 0, 1],
                                [l*(l+1)/r**2 - eps_v**2, -V(r, eps_phi), 0, 0],
                                [-V(r, eps_phi), l*(l+1)/r**2 + eps_delta**2 - eps_v**2, 0, 0]
                                ])
        new_X = np.dot(equation, X)
        return new_X

    #The derivative of psi11
    def extremum1(r, X):
        return X[2]

    #The derivative of psi22
    def extremum2(r, X):
        return X[3]

    #Solving the equation
    time0 = time.time()
    sol1 = solve_ivp(fun, (r_min, r_max), y0 = np.array([1, 0, (l+1)/r_min, 0]), t_eval = r_tab, events = (extremum1, extremum2), method = 'Radau', rtol = 10**(-numerical_accuracy))
    sol2 = solve_ivp(fun, (r_min, r_max), y0 = np.array([0, 1, 0, (l+1)/r_min]), t_eval = r_tab, events = (extremum1, extremum2), method = 'Radau', rtol = 10**(-numerical_accuracy))
    time1 = time.time()


    ##Printing out time of calculation and end of integration status
    if print_out:
        print("Time of calculation: ", time1 - time0, "s")
    if (below_threshold) and (sol1.status != 0 or print_out):
        print('Integration status: ', sol1.status)
    if (not below_threshold) and (sol1.status != 0 or sol2.status!= 0 or print_out):
        print('Integration status: ', sol1.status, " ", sol2.status)

    #Finding the optimal initial value for below_threshold solution
    ###Scipy bisect method
    if below_threshold:
        def chi(ratio):
            sol = solve_ivp(fun, (r_min, r_max),
                            y0 = ratio * np.array([1, 0, (l+1)/r_min, 0]) - np.array([0, 1, 0, (l+1)/r_min]),
                            t_eval = r_tab, events = (extremum1, extremum2), method = 'Radau', rtol = 10**(-numerical_accuracy))
            return sol.y[1,-1]

        ratio_lower, ratio_upper = np.longdouble(0.9 * sol2.y[1,-1] / sol1.y[1,-1]), np.longdouble(1.1 * sol2.y[1,-1] / sol1.y[1,-1])
        ratio_optimal = bisect(chi, ratio_lower, ratio_upper)
        if print_out:
            print('Optimal initial values ratio: ', ratio_optimal)
        damping = False
        sol1 = solve_ivp(fun, (r_min, r_max),
                        y0 = ratio_optimal * np.array([1, 0, (l+1)/r_min, 0]) - np.array([0, 1, 0, (l+1)/r_min]),
                        t_eval = r_tab, events = (extremum1, extremum2), method = 'Radau', rtol = 10**(-numerical_accuracy))

    ###My bisect method
    # if below_threshold:
    #     ratio_lower = np.longdouble(0.9 * sol2.y[1,-1] / sol1.y[1,-1])
    #     sol = solve_ivp(fun, (r_min, r_max),
    #                     y0 = ratio_lower * np.array([1, 0, (l+1)/r_min, 0]) - np.array([0, 1, 0, (l+1)/r_min]),
    #                     t_eval = r_tab, events = (extremum1, extremum2), method = 'Radau', rtol = 10**(-numerical_accuracy))
    #     chi_lower = sol.y[1,-1]
    #
    #     ratio_upper = np.longdouble(1.1 * sol2.y[1,-1] / sol1.y[1,-1])
    #     sol = solve_ivp(fun, (r_min, r_max),
    #                     y0 = ratio_upper * np.array([1, 0, (l+1)/r_min, 0]) - np.array([0, 1, 0, (l+1)/r_min]),
    #                     t_eval = r_tab, events = (extremum1, extremum2), method = 'Radau', rtol = 10**(-numerical_accuracy))
    #     chi_upper = sol.y[1,-1]
    #
    #     sol = solve_ivp(fun, (r_min, r_max),
    #                     y0 = (ratio_lower + ratio_upper)/2 * np.array([1, 0, (l+1)/r_min, 0]) - np.array([0, 1, 0, (l+1)/r_min]),
    #                     t_eval = r_tab, events = (extremum1, extremum2), method = 'Radau', rtol = 10**(-numerical_accuracy))
    #     chi = sol.y[1,-1]
    #
    #     while np.abs(chi) > 0.01:
    #         if chi/chi_lower > 0:
    #             ratio_lower = (ratio_lower + ratio_upper)/2
    #             sol = solve_ivp(fun, (r_min, r_max),
    #                             y0 = ratio_lower * np.array([1, 0, (l+1)/r_min, 0]) - np.array([0, 1, 0, (l+1)/r_min]),
    #                             t_eval = r_tab, events = (extremum1, extremum2), method = 'Radau', rtol = 10**(-numerical_accuracy))
    #             chi_lower = sol.y[1,-1]
    #         else:
    #             ratio_upper = (ratio_lower + ratio_upper)/2
    #             sol = solve_ivp(fun, (r_min, r_max),
    #                             y0 = ratio_upper * np.array([1, 0, (l+1)/r_min, 0]) - np.array([0, 1, 0, (l+1)/r_min]),
    #                             t_eval = r_tab, events = (extremum1, extremum2), method = 'Radau', rtol = 10**(-numerical_accuracy))
    #             chi_upper = sol.y[1,-1]
    #         sol = solve_ivp(fun, (r_min, r_max),
    #                         y0 = (ratio_upper + ratio_lower)/2 * np.array([1, 0, (l+1)/r_min, 0]) - np.array([0, 1, 0, (l+1)/r_min]),
    #                         t_eval = r_tab, events = (extremum1, extremum2), method = 'Radau', rtol = 10**(-numerical_accuracy))
    #         chi = sol.y[1,-1]
    #         if print_out:
    #             print()
    #             print('ratio_lower: {} ratio_upper: {}'.format(ratio_lower, ratio_upper))
    #             print("chi_lower: {:.5e} chi: {:.5e} chi_upper: {:.5e}".format(chi_lower, chi, chi_upper))
    #
    #     ratio_optimal = (ratio_lower + ratio_upper)/2
    #
    #     sol1 = solve_ivp(fun, (r_min, r_max),
    #                     y0 = ratio_optimal * np.array([1, 0, (l+1)/r_min, 0]) - np.array([0, 1, 0, (l+1)/r_min]),
    #                     t_eval = r_tab, events = (extremum1, extremum2), method = 'Radau', rtol = 10**(-numerical_accuracy))


    ##Notation: first digit signifies the initial value (either non-zero 11 or non-zero 22), the second describes the state (either 11 or 22),
    ##that is the first index stands for i and the second for n
    r = r_tab

    ##Substituting in the solutions

    psi11 = sol1.y[0]
    Psi11 = sol1.y[0]/r
    r_extrema_11 = sol1.t_events[0]

    psi12 = sol1.y[1]
    Psi12 = sol1.y[1]/r
    r_extrema_12 = sol1.t_events[1]

    if not below_threshold:

        psi21 = sol2.y[0]
        Psi21 = sol2.y[0]/r
        r_extrema_21 = sol2.t_events[0]

        psi22 = sol2.y[1]
        Psi22 = sol2.y[1]/r
        r_extrema_22 = sol2.t_events[1]

    #Plotting the solutions
    if plot_sol:
        directory_path = os.getcwd()
        try:
            os.mkdir(directory_path + "/eps_v={}_eps_delta={}_eps_phi={}_l={}".format(eps_v, eps_delta, eps_phi, l))
        except:
            pass

        fig, ax = plt.subplots()
        plt.plot(r, psi11)
        ax.set_xscale('log', base=10)
        plt.savefig('eps_v={}_eps_delta={}_eps_phi={}_l={}/solution(11_init_11_state).png'. format(eps_v, eps_delta, eps_phi, l))
        plt.close()

        fig, ax = plt.subplots()
        plt.plot(r, psi12)
        ax.set_xscale('log', base=10)
        plt.savefig('eps_v={}_eps_delta={}_eps_phi={}_l={}/solution(11_init_22_state).png'. format(eps_v, eps_delta, eps_phi, l))
        plt.close()

        if not below_threshold:

            fig, ax = plt.subplots()
            plt.plot(r, psi21)
            ax.set_xscale('log', base=10)
            plt.savefig('eps_v={}_eps_delta={}_eps_phi={}_l={}/solution(22_init_11_state).png'. format(eps_v, eps_delta, eps_phi, l))
            plt.close()

            fig, ax = plt.subplots()
            plt.plot(r, psi22)
            ax.set_xscale('log', base=10)
            plt.savefig('eps_v={}_eps_delta={}_eps_phi={}_l={}/solution(22_init_22_state).png'. format(eps_v, eps_delta, eps_phi, l))
            plt.close()

    #Fitting the Bessel solution
    def bessel1(r, a, delta):
        k = eps_v
        return a*np.sin(k*r - 1/2*l*np.pi + delta)

    def bessel2(r, a, delta):
        k = np.sqrt(eps_v**2 - eps_delta**2)
        return a*np.sin(k*r - 1/2*l*np.pi + delta)

    ##Improving fitting by imposing bounds should help the algorithm, but it doesn't seem to work
    ##Fitting to last 5 peaks of the oscillating solution
    i11 = np.argwhere(r_tab >= r_extrema_11[-5])[0, 0]
    (a11, delta11), cov11 = curve_fit(bessel1, r[i11:], psi11[i11:], p0 = (psi11[i11], 0)) # bounds = ([-np.max(psi11), -2*np.pi], [np.max(psi11), 2*np.pi])
    delta11 = delta11%(2*np.pi)

    if not below_threshold:

        i12 = np.argwhere(r_tab >= r_extrema_12[-5])[0, 0]
        (a12, delta12), cov12 = curve_fit(bessel2, r[i12:], psi12[i12:], p0 = (psi12[i12], 0))
        delta12 = delta12%(2*np.pi)

        i21 = np.argwhere(r_tab >= r_extrema_21[-5])[0, 0]
        (a21, delta21), cov21 = curve_fit(bessel1, r[i21:], psi21[i21:], p0 = (psi21[i21], 0))
        delta21 = delta21%(2*np.pi)

        i22 = np.argwhere(r_tab >= r_extrema_22[-5])[0, 0]
        (a22, delta22), cov22 = curve_fit(bessel2, r[i22:], psi22[i22:], p0 = (psi22[i22], 0))
        delta22 = delta22%(2*np.pi)

    else:
        a12, delta12 = 0, 0
        a21, delta21 = 0, 0
        a22, delta22 = 1, 0

    #Printing out the solution
    if print_out:
        print("a11, delta11: ", a11, delta11)
        print("a12, delta12: ", a12, delta12)
        print("a21, delta21: ", a21, delta21)
        print("a22, delta22: ", a22, delta22)

    #Plotting the solutions along with fitted Bessel solutions
    if plot_fit:
        psi11_fit = np.zeros(N-i11)
        for i in range(i11, N):
            psi11_fit[i-i11] = bessel1(r[i], a11, delta11)
        if not below_threshold:
            psi12_fit = np.zeros(N-i12)
            for i in range(i12, N):
                psi12_fit[i-i12] = bessel2(r[i], a12, delta12)
            psi21_fit = np.zeros(N-i21)
            for i in range(i21, N):
                psi21_fit[i-i21] = bessel1(r[i], a21, delta21)
            psi22_fit = np.zeros(N-i22)
            for i in range(i22, N):
                psi22_fit[i-i22] = bessel2(r[i], a22, delta22)

        directory_path = os.getcwd()
        try:
            os.mkdir(directory_path + "/eps_v={}_eps_delta={}_eps_phi={}_l={}".format(eps_v, eps_delta, eps_phi, l))
        except:
            pass

        fig, ax = plt.subplots()
        plt.plot(r[i11:], psi11[i11:]/np.abs(a11))
        plt.plot(r[i11:], psi11_fit[:]/np.abs(a11))
        plt.savefig('eps_v={}_eps_delta={}_eps_phi={}_l={}/solution_fit(11_init_11_state).png'. format(eps_v, eps_delta, eps_phi, l))
        plt.close()

        if not below_threshold:

            fig, ax = plt.subplots()
            plt.plot(r[i12:], psi12[i12:]/np.abs(a12))
            plt.plot(r[i12:], psi12_fit/np.abs(a12))
            plt.savefig('eps_v={}_eps_delta={}_eps_phi={}_l={}/solution_fit(11_init_22_state).png'. format(eps_v, eps_delta, eps_phi, l))
            plt.close()

            fig, ax = plt.subplots()
            plt.plot(r[i21:], psi21[i21:]/np.abs(a21))
            plt.plot(r[i21:], psi21_fit/np.abs(a21))
            plt.savefig('eps_v={}_eps_delta={}_eps_phi={}_l={}/solution_fit(22_init_11_state).png'. format(eps_v, eps_delta, eps_phi, l))
            plt.close()

            fig, ax = plt.subplots()
            plt.plot(r[i22:], psi22[i22:]/np.abs(a22))
            plt.plot(r[i22:], psi22_fit/np.abs(a22))
            plt.savefig('eps_v={}_eps_delta={}_eps_phi={}_l={}/solution_fit(22_init_22_state).png'. format(eps_v, eps_delta, eps_phi, l))
            plt.close()

    #Normalising the solution
    a12 /= a11
    a11 /= a11
    a21 /= a22
    a22 /= a22

    #Calculating the S matrix from the solution
    F_tilde = np.array([
                [a11 * np.exp(-1j*delta11), a21 * np.exp(-1j*delta21)],
                [a12 * np.exp(-1j*delta12), a22 * np.exp(-1j*delta22)]
                ])
    if not below_threshold:
        k_0 = eps_v
        k_1 = np.sqrt(eps_v**2 - eps_delta**2)
        C = np.sqrt(k_1/k_0)
    else:
        k_0 = eps_v
        C = 1
    F = np.dot(np.array([[1, 0], [0, C]]), F_tilde)
    S = np.dot(np.conjugate(F), np.linalg.inv(F))
    if print_out:
        print("F:")
        print(F)
        print("S:")
        print(S)

    ##Below threshold the phase can be retrieved from S with sqrt(conj(S[0,0]))

    #Finding the numerical error (by checking the unitarity of S)
    numerical_error = np.linalg.norm(np.eye(2) - np.dot(S,S.conj().T))
    if print_out or numerical_error > 10**(2-numerical_accuracy):
        print("Distance from unitarity: ", numerical_error)

    # Calculating the scattering cross-section
    if not below_threshold:
        sigma_l = np.pi / np.array([[k_0, k_1], [k_0, k_1]])**2 * (2*l+1) * np.abs(S - np.identity(2))**2
    else:
        sigma_l = np.pi / np.array([[k_0, np.inf], [np.inf, np.inf]])**2 * (2*l+1) * np.abs(S - np.identity(2))**2

    if print_out:
        print("sigma_l:")
        print(sigma_l)

    ### TASK: Sommerfeld enhancement factor
    # (l+1) derivative of psi at r_min = (l+1)! * psi(r_min)/r_min**(l+1)
    # Insert it into (2.10) where M_l = F_l and Gamma is [[2,0], [0,0]] and C is c/k, where c is initial condition vector and k is the momentum vector
    #                   -> from Sommerfeld enhancement paper

    # Calculating the Sommerfeld enhancement factor
    ## For 11 initial condition
    c = np.array([[1, 0], [0,1]])
    k = np.array([k_0, k_1])
    Gamma = np.array([[2,0], [0,0]])
    chi = np.array([[psi11[0], psi12[0]], [psi21[0], psi22[0]]])
    numerator = factorial(l+1)*np.linalg.multi_dot([np.linalg.multi_dot([chi/r_min**(l+1), np.linalg.inv(F), c/k]).conj().T, Gamma, np.linalg.multi_dot([chi/r_min**(l+1), np.linalg.inv(F), c/k])])
    denominator1, denominator2 = 0, 0
    for i in [0,1]:
        for j in [0,1]:
            denominator1 += (k[i]*k[j])**l * c[0,i] * Gamma[i,j] * c[0,j]
            denominator2 += (k[i]*k[j])**l * c[1,i] * Gamma[i,j] * c[1,j]
    S_1 = ((factorial2(2*l+1)/factorial(l+1))**2 * numerator/denominator1)[0,0]
    S_2 = ((factorial2(2*l+1)/factorial(l+1))**2 * numerator/denominator2)[1,1]

    S_0 = np.abs((factorial2(2*l+1) * factorial(l+1)*psi11[0]/r_min**(l+1)) / (factorial(l+1)*k_0**(l+1)))**2

    if print_out:
        print("S_0:")
        print(S_0)
        print("S_1:")
        print(S_1)
        print("S_2:")
        print(S_2)

    return S, sigma_l, S_1

solution(eps_v=0.2, eps_delta=0.1, eps_phi=0.2, print_out=True, plot_sol=True, plot_fit=True)

