####################################
# Author: Mateusz Kulejewski, 2023 #
####################################

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

def solution(
            eps_v, eps_delta, eps_phi,
            V = Yukawa, l = 0,
            print_out=False, plot_sol=False, plot_fit=False, plot_chi_sq=False,
            damping = False, numerical_accuracy = 4, r_max = None, update_accuracy=True):

    if print_out:
        print('eps_v: ', eps_v, ' eps_delta: ', eps_delta, ' eps_phi: ', eps_phi)
        print('r_max: ', r_max, ' numerical accuracy: ', numerical_accuracy)
    below_threshold = (eps_v <= eps_delta)
    if print_out:
        print("Below threshold: ", below_threshold)

    r_min = 1/100
    if r_max is None:
        r_max = np.max([l*100/eps_v, 100/eps_phi])
        if below_threshold:
            r_max /= 100
    N = 100000
    r_tab = np.geomspace(r_min, r_max, N)
    if below_threshold:
        eps_damping = np.min((eps_v, np.sqrt(eps_delta**2 - eps_v**2)))/100

    #Schrodinger equation
    def fun(r, X):
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
    t = time.time()
    sol1 = solve_ivp(fun, (r_min, r_max),
                    y0 = np.array([1, 0, (l+1)/r_min, 0]),
                    t_eval = r_tab, events = (extremum1, extremum2), method = 'Radau', rtol = 10**(-numerical_accuracy))
    sol2 = solve_ivp(fun, (r_min, r_max),
                    y0 = np.array([0, 1, 0, (l+1)/r_min]),
                    t_eval = r_tab, events = (extremum1, extremum2), method = 'Radau', rtol = 10**(-numerical_accuracy))

    ##Printing out time of calculation and end of integration status
    if print_out:
        print("Time of integration: ", time.time() - t, "s")
    if (below_threshold) and (sol1.status != 0 or print_out):
        print('Integration status: ', sol1.status)
    if (not below_threshold) and (sol1.status != 0 or sol2.status!= 0 or print_out):
        print('Integration status: ', sol1.status, " ", sol2.status)

    #Finding the optimal initial value for below_threshold solution
    if below_threshold:
        #Damped Schrodinger equation
        def fun_damp(r, X):
            equation = np.array([
                                [0, 0, 1, 0],
                                [0, 0, 0, np.exp(-eps_damping*r**2)*1],
                                [l*(l+1)/r**2 - eps_v**2, -V(r, eps_phi), 0, 0],
                                [np.exp(-eps_damping*r**2)*(-V(r, eps_phi)), np.exp(-eps_damping*r**2)*(l*(l+1)/r**2 + eps_delta**2 - eps_v**2), 0, 0]
                                ])
            new_X = np.dot(equation, X)
            return new_X

        t = time.time()
        def chi(ratio):
            sol = solve_ivp(fun, (r_min, r_max),
                            y0 = ratio * np.array([1, 0, (l+1)/r_min, 0]) - np.array([0, 1, 0, (l+1)/r_min]),
                            t_eval = r_tab, events = (extremum1, extremum2), method = 'Radau', rtol = 10**(-numerical_accuracy))
            return sol.y[1,-1]

        ratio_lower, ratio_upper = np.longdouble(0.9 * sol2.y[1,-1] / sol1.y[1,-1]), np.longdouble(1.1 * sol2.y[1,-1] / sol1.y[1,-1])
        ratio_optimal = bisect(chi, ratio_lower, ratio_upper)
        if print_out:
            print('Optimal initial values ratio: ', ratio_optimal)
            print("Time of finding optimal initial values: ", time.time() - t, "s")
        t = time.time()
        if damping:
            sol1 = solve_ivp(fun_damp, (r_min, r_max),
                            y0 = ratio_optimal * np.array([1, 0, (l+1)/r_min, 0]) - np.array([0, 1, 0, (l+1)/r_min]),
                            t_eval = r_tab, events = (extremum1, extremum2), method = 'Radau', rtol = 10**(-numerical_accuracy))
        else:
            sol1 = solve_ivp(fun, (r_min, r_max),
                            y0 = ratio_optimal * np.array([1, 0, (l+1)/r_min, 0]) - np.array([0, 1, 0, (l+1)/r_min]),
                            t_eval = r_tab, events = (extremum1, extremum2), method = 'Radau', rtol = 10**(-numerical_accuracy))
        print("Time of integration: ", time.time() - t, "s")

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
    ##Fitting to last 5 zeroes of the oscillating solution
    if update_accuracy:
        try:
            i11 = np.argwhere(r_tab >= r_extrema_11[-5])[0, 0]
        except IndexError:
            if print_out:
                print()
                print('No oscillating solution found, increasing maximal distance...')
                print()
            return solution(eps_v = eps_v, eps_delta = eps_delta, eps_phi = eps_phi, V = V, l = l, damping = damping, print_out=print_out, plot_sol=plot_sol, plot_fit=plot_fit, plot_chi_sq=plot_chi_sq, numerical_accuracy = numerical_accuracy, r_max = 2*r_max)
    else:
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
    # a12 /= a11
    # a11 /= a11
    # a21 /= a22
    # a22 /= a22

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
        k_1 = np.inf
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
    if print_out:
        print("Distance from unitarity: ", numerical_error)
    if numerical_error > 10**(-2) and update_accuracy:
        if print_out:
            print()
            print('S matrix not unitary, increasing numerical accuracy...')
            print()
        return solution(eps_v = eps_v, eps_delta = eps_delta, eps_phi = eps_phi, V = V, l = l, damping = damping, print_out=print_out, plot_sol=plot_sol, plot_fit=plot_fit, plot_chi_sq=plot_chi_sq, numerical_accuracy = numerical_accuracy + 1, r_max = r_max)

    # Calculating the scattering cross-section
    ### sigma_l = np.pi / k_i**2 * (2*l + 1) * np.abs( S[f, i] - np.id(2)[f,i])**2 for transition from state i to state f

    sigma_l = np.pi / np.array([[k_0, k_1], [k_0, k_1]])**2 * (2*l+1) * np.abs(S - np.identity(2))**2

    if print_out:
        print("sigma_l:")
        print(sigma_l)

    ### TASK: Sommerfeld enhancement factor
    # (l+1) derivative of psi at r_min = (l+1)! * psi(r_min)/r_min**(l+1)
    # Insert it into (2.10) where M_l = F_l and Gamma is [[2,0], [0,0]] and C is c/k, where c is initial condition vector and k is the momentum vector
    #                   -> from Sommerfeld enhancement paper

    # Calculating the Sommerfeld enhancement factor
    if below_threshold:
        c = np.array([[1, 0], [0,0]])
        k = np.array([k_0, 0])
        Gamma = np.array([[1,1], [1,1]])
        chi = np.array([[psi11[0], 0, 0, 0]])

    if not below_threshold:
        c = np.array([[1, 0], [0,1]])
        k = np.array([k_0, k_1])
        Gamma = np.array([[1,1], [1,1]])
        chi = np.array([[psi11[0], psi21[0]], [psi12[0], psi22[0]]])

    numerator = factorial(l+1)*np.linalg.multi_dot([np.linalg.multi_dot([chi/r_min**(l+1), np.linalg.inv(F), c/k]).conj().T, Gamma, np.linalg.multi_dot([chi/r_min**(l+1), np.linalg.inv(F), c/k])])
    denominator1 = 0
    for i in [0,1]:
        for j in [0,1]:
            denominator1 += (k[i]*k[j])**l * c[0,i] * Gamma[i,j] * c[0,j]
    S_1 = ((factorial2(2*l+1)/factorial(l+1))**2 * numerator/denominator1)[0,0]

    if print_out:
        print("S_1:")
        print(S_1)

    return S, sigma_l, S_1


def save_S(eps_v_min, eps_v_max, eps_v_N, eps_delta_min, eps_delta_max, eps_delta_N, eps_phi_min, eps_phi_max, eps_phi_N):
    eps_v_tab = np.linspace(eps_v_min, eps_v_max, eps_v_N)
    eps_delta_tab = np.linspace(eps_delta_min, eps_delta_max, eps_delta_N)
    eps_phi_tab = np.linspace(eps_phi_min, eps_phi_max, eps_phi_N)
    data = np.zeros((eps_v_N*eps_delta_N*eps_phi_N, 7), dtype='cdouble')
    for i in tqdm(range(eps_phi_N), leave=False):
        for j in tqdm(range(eps_delta_N), leave=False):
            for k in tqdm(range(eps_v_N), leave=False):
    # for i in range(eps_phi_N):
    #     for j in range(eps_delta_N):
    #         for k in range(eps_v_N):
                eps_phi = eps_phi_tab[i]
                eps_delta = eps_delta_tab[j]
                eps_v = eps_v_tab[k]
                try:
                    S = solution(eps_v=eps_v, eps_delta=eps_delta, eps_phi=eps_phi)#, print_out=True)
                except:
                    print('Solution failed for eps_v={}, eps_delta={}, eps_phi={}'.format(eps_v, eps_delta, eps_phi))
                # print(i*eps_delta_N*eps_v_N + j*eps_v_N + k)
                data[i*eps_delta_N*eps_v_N + j*eps_v_N + k] = np.array([eps_v, eps_delta, eps_phi, S[0,0], S[0,1], S[1,0], S[1,1]])
                # print(data[0:i*eps_delta_N*eps_v_N + j*eps_v_N + k + 1,:])
                # if i*eps_delta_N*eps_v_N + j*eps_v_N + k > 0:
                np.savetxt(fname = 'sols.txt', X = data[0:i*eps_delta_N*eps_v_N + j*eps_v_N + k + 1,:],
                            fmt= ['%10.5f + %10.5fj', '%10.5f + %10.5fj', '%10.5f + %10.5fj', '%10.5f + %10.5fj', '%10.5f + %10.5fj', '%10.5f + %10.5fj', '%10.5f + %10.5fj'],
                            header = ' eps_v                    eps_delta                eps_phi                  S11                      S12                      S21                      S22',
                            footer = ' eps_v                    eps_delta                eps_phi                  S11                      S12                      S21                      S22')

    return
# save_S(0.1, 0.5, 25, 0.1, 0.25, 5, 0.1, 0.5, 2)

def plot_S_of_eps_v(eps_v_min, eps_v_max, eps_v_N):
    eps_v_tab = np.linspace(eps_v_min, eps_v_max, eps_v_N)
    S_tab = np.zeros((eps_v_N, 2, 2), dtype='cdouble')
    for i in tqdm(range(eps_v_N)):
        eps_v = eps_v_tab[i]
        S = solution(eps_v=eps_v, eps_delta=eps_delta, eps_phi=eps_phi)
        S_tab[i] = S

    fig, ax = plt.subplots()
    plt.plot(eps_v_tab, np.abs(S_tab[:,0,0]))
    plt.plot(eps_v_tab, np.real(S_tab[:,0,0]))
    plt.plot(eps_v_tab, np.imag(S_tab[:,0,0]))
    plt.xlim(eps_v_min, eps_v_max)
    plt.ylim(-1, 1)
    plt.xlabel(r'$\varepsilon_v$')
    plt.ylabel(r'$S_{11}$')
    plt.legend([r'Abs($S_{11}$)', r'Re($S_{11}$)', r'Im($S_{11}$)'])
    plt.savefig('S_11_eps_delta={}_eps_phi={}_l={}.png'. format(eps_delta, eps_phi, l))
    plt.close()

    fig, ax = plt.subplots()
    plt.plot(eps_v_tab, np.abs(S_tab[:,0,1]))
    plt.plot(eps_v_tab, np.real(S_tab[:,0,1]))
    plt.plot(eps_v_tab, np.imag(S_tab[:,0,1]))
    plt.xlim(eps_v_min, eps_v_max)
    plt.ylim(-1, 1)
    plt.xlabel(r'$\varepsilon_v$')
    plt.ylabel(r'$S_{12}$')
    plt.legend([r'Abs($S_{12}$)', r'Re($S_{12}$)', r'Im($S_{12}$)'])
    plt.savefig('S_12_eps_delta={}_eps_phi={}_l={}.png'. format(eps_delta, eps_phi, l))
    plt.close()

    fig, ax = plt.subplots()
    plt.plot(eps_v_tab, np.abs(S_tab[:,1,0]))
    plt.plot(eps_v_tab, np.real(S_tab[:,1,0]))
    plt.plot(eps_v_tab, np.imag(S_tab[:,1,0]))
    plt.xlim(eps_v_min, eps_v_max)
    plt.ylim(-1, 1)
    plt.xlabel(r'$\varepsilon_v$')
    plt.ylabel(r'$S_{21}$')
    plt.legend([r'Abs($S_{21}$)', r'Re($S_{21}$)', r'Im($S_{21}$)'])
    plt.savefig('S_21_eps_delta={}_eps_phi={}_l={}.png'. format(eps_delta, eps_phi, l))
    plt.close()

    fig, ax = plt.subplots()
    plt.plot(eps_v_tab, np.abs(S_tab[:,1,1]))
    plt.plot(eps_v_tab, np.real(S_tab[:,1,1]))
    plt.plot(eps_v_tab, np.imag(S_tab[:,1,1]))
    plt.xlim(eps_v_min, eps_v_max)
    plt.ylim(-1, 1)
    plt.xlabel(r'$\varepsilon_v$')
    plt.ylabel(r'$S_{22}$')
    plt.legend([r'Abs($S_{22}$)', r'Re($S_{22}$)', r'Im($S_{22}$)'])
    plt.savefig('S_22_eps_delta={}_eps_phi={}_l={}.png'. format(eps_delta, eps_phi, l))
    plt.close()

    return

def plot_sigma_l_3D(eps_delta, eps_v_min, eps_v_max, eps_phi_min, eps_phi_max, N):
    sigma_l_tab = np.zeros((N, N, 4))
    eps_v_tab = np.geomspace(eps_v_min, eps_v_max, N)
    eps_phi_tab = np.geomspace(eps_phi_min, eps_phi_max, N)
    for i in tqdm(range(N), leave=False):
        eps_phi = eps_phi_tab[i]
        for j in tqdm(range(N), leave=False):
            eps_v = eps_v_tab[j]
            if eps_phi < eps_v:
                sigma_l_tab[i,j] = np.zeros(4)
            else:
                sigma_l = solution(eps_v = eps_v, eps_delta = eps_delta, eps_phi = eps_phi)[1]
                sigma_l_tab[i,j,0] = sigma_l[0,0] # Elastic scattering in the ground state
                sigma_l_tab[i,j,1] = sigma_l[0,1] # Upscattering
                sigma_l_tab[i,j,2] = sigma_l[1,0] # Downscattering
                sigma_l_tab[i,j,3] = sigma_l[1,1] # Elastic scattering in the excited state

    plt.imshow(np.log10(sigma_l_tab[:,:,0]), origin = 'lower')
    plt.title('log(sigma) in elastic scattering in the ground state for eps_delta={}'.format(eps_delta))
    plt.xlabel('log(eps_v)')
    plt.ylabel('log(eps_phi)')
    plt.xticks(np.linspace(0, 1, 5), labels = np.log10(np.linspace(eps_v_min, eps_v_max, 5)))
    plt.yticks(np.linspace(0, 1, 5), labels = np.log10(np.linspace(eps_phi_min, eps_phi_max, 5)))
    plt.savefig('elastic_scattering_ground_state_eps_delta={}.png'.format(eps_delta))
    plt.close()

    plt.imshow(np.log10(sigma_l_tab[:,:,1]), origin = 'lower')
    plt.title('log(sigma) in upscattering for eps_delta={}'.format(eps_delta))
    plt.xlabel('log(eps_v)')
    plt.ylabel('log(eps_phi)')
    plt.xticks(np.linspace(0, 1, 5), labels = np.log10(np.linspace(eps_v_min, eps_v_max, 5)))
    plt.yticks(np.linspace(0, 1, 5), labels = np.log10(np.linspace(eps_phi_min, eps_phi_max, 5)))
    plt.savefig('upscattering_eps_delta={}.png'.format(eps_delta))
    plt.close()

    plt.imshow(np.log10(sigma_l_tab[:,:,2]), origin = 'lower')
    plt.title('log(sigma) in downscattering for eps_delta={}'.format(eps_delta))
    plt.xlabel('log(eps_v)')
    plt.ylabel('log(eps_phi)')
    plt.xticks(np.linspace(0, 1, 5), labels = np.log10(np.linspace(eps_v_min, eps_v_max, 5)))
    plt.yticks(np.linspace(0, 1, 5), labels = np.log10(np.linspace(eps_phi_min, eps_phi_max, 5)))
    plt.savefig('downscattering_eps_delta={}.png'.format(eps_delta))
    plt.close()

    plt.imshow(np.log10(sigma_l_tab[:,:,3]), origin = 'lower')
    plt.title('log(sigma) in elastic scattering in excited state for eps_delta={}'.format(eps_delta))
    plt.xlabel('log(eps_v)')
    plt.ylabel('log(eps_phi)')
    plt.xticks(np.linspace(0, 1, 5), labels = np.log10(np.linspace(eps_v_min, eps_v_max, 5)))
    plt.yticks(np.linspace(0, 1, 5), labels = np.log10(np.linspace(eps_phi_min, eps_phi_max, 5)))
    plt.savefig('elastic_scattering_excited_state_eps_delta={}.png'.format(eps_delta))
    plt.close()

def plot_2D(eps_delta, eps_v, eps_phi_min, eps_phi_max, N):
    sigma_l_tab = np.zeros((N, 4))
    S_tab = np.zeros(N)
    eps_phi_tab = np.geomspace(eps_phi_min, eps_phi_max, N)
    for i in tqdm(range(N)):
        eps_phi = eps_phi_tab[i]
        sol = solution(eps_v = eps_v, eps_delta = eps_delta, eps_phi = eps_phi, print_out = True)
        sigma_l, S = sol[1], sol[2]
        sigma_l_tab[i,0] = sigma_l[0,0] # Elastic scattering in the ground state
        sigma_l_tab[i,1] = sigma_l[0,1] # Upscattering
        sigma_l_tab[i,2] = sigma_l[1,0] # Downscattering
        sigma_l_tab[i,3] = sigma_l[1,1] # Elastic scattering in the excited state
        S_tab[i] = S

    plt.plot(eps_phi_tab, sigma_l_tab[:,0])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\epsilon_\phi$')
    plt.ylabel(r'$\sigma$')
    plt.savefig('sigma_eps_delta={}_eps_v={}_elastic_ground.png'.format(eps_delta, eps_v))
    # plt.show()
    plt.close()

    plt.plot(eps_phi_tab, sigma_l_tab[:,1])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\epsilon_\phi$')
    plt.ylabel(r'$\sigma$')
    plt.savefig('sigma_eps_delta={}_eps_v={}_upscattering.png'.format(eps_delta, eps_v))
    # plt.show()
    plt.close()

    plt.plot(eps_phi_tab, sigma_l_tab[:,2])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\epsilon_\phi$')
    plt.ylabel(r'$\sigma$')
    plt.savefig('sigma_eps_delta={}_eps_v={}_downscattering.png'.format(eps_delta, eps_v))
    # plt.show()
    plt.close()

    plt.plot(eps_phi_tab, sigma_l_tab[:,3])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\epsilon_\phi$')
    plt.ylabel(r'$\sigma$')
    plt.savefig('sigma_eps_delta={}_eps_v={}_elastic_excited.png'.format(eps_delta, eps_v))
    # plt.show()
    plt.close()

    plt.plot(eps_phi_tab, S_tab)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\epsilon_\phi$')
    plt.ylabel('S')
    plt.savefig('S_eps_delta={}_eps_v={}.png'.format(eps_delta, eps_v))
    # plt.show()
    plt.close()


# plot_sigma_l_3D(eps_delta = 0.01, eps_v_min = 10**(-2), eps_v_max = 1, eps_phi_min = 10**(-2), eps_phi_max = 1, N = 5)
solution(eps_v=0.01, eps_delta=0.1, eps_phi=0.01, print_out=True, plot_sol=True, plot_fit=True)
# plot_2D(eps_delta = 0.01, eps_v = 0.001, eps_phi_min = 10**(-2), eps_phi_max = 10**(0), N = 50)

### DONE TASK: S matrix as a function of eps_v for given eps_phi, eps_delta and l
### Save to csv file: eps_v, eps_delta, eps_phi, S11, S12, S21, S22
### Find the phaseshift of 11->11 below threshold

### DONE TASK: to eliminate the exponential growth in 22 below threshold, minimise the function \chi**2 (B/A) = |psi22(r_max)|**2 for B/A near to B = psi2.y[1,-1], A = psi1.y[1,-1]

### sigma_l = np.pi / k_i**2 * (2*l + 1) * np.abs( S[f, i] - np.id(2)[f,i])**2 for transition from state i to state f

### exponential divergence of 22 state below threshold has negligible influence on 11 state as long as eps_phi > np.sqrt(eps_delta**2 - eps_v**2)

### check the impact of damping factor on the phaseshift
# It does change, delta ~ 0.03
# But it does not change after taking optimal initial values

### DONE TASK: introduce damping only after solving for the ratio initial value DONE

### DONE TASK: compute the sigma_l

### TASK: Sommerfeld enhancement factor
# (l+1) derivative of psi at r_min = (l+1)! * psi(r_min)/r_min**(l+1)
# Insert it into (2.10) where M_l = F_l and Gamma is [[2,0], [0,0]] and C is c/k, where c is initial condition vector and k is the momentum vector
#                   -> from Sommerfeld enhancement paper

### TASK: Reproduce Figures 6 from self-scattering paper
### Perhaps consider other l

### TASK: Plot Figures 3 & 4 from Sommerfeld enhancement paper
### Think what can be wrong

### IMPROVEMENT: Bisecting the maximal distance for too large r_max solutions

### BETTER IMPROVEMENT for l>0: r_max = 10/eps_phi, take fitting from (11) of 2303.17961

### TASK: Sommerfeld enhancement factor below threshold: express the special initial value solution in the basis of 11 and 22 states
