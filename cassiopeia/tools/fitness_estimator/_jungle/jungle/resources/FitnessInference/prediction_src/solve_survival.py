#########################################################################################
#
# author: Richard Neher
# email: richard.neher@tuebingen.mpg.de
#
# Reference: Richard A. Neher, Colin A Russell, Boris I Shraiman.
#            "Predicting evolution from the shape of genealogical trees"
#
#########################################################################################
#
# solve_survival.py
# provides a class called survival_gen_func that solves for the generating function
# of a branching process asuming diffusive changes in fitness over time. In addition
# it provides numeric solutions for the branch propagator, which is no longer time
# invariant and requires initial and final time due to the non-sampling constraint
# discussed in the above manuscript.
#
# the implementation uses a simple fourth order runge kutta integration
#
#########################################################################################


import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d

non_negativity_cutoff = 1e-20


def integrate_rk4(df, f0, T, dt, args):
    """
    self made fixed time step rk4 integration
    """
    sol = np.zeros([len(T)] + list(f0.shape))
    sol[0] = f0
    f = f0.copy()
    t = T[0]
    for ti, tnext in enumerate(T[1:]):
        while t < tnext:
            h = min(dt, tnext - t)
            k1 = df(f, t, *args)
            k2 = df(f + 0.5 * h * k1, t + 0.5 * h, *args)
            k3 = df(f + 0.5 * h * k2, t + 0.5 * h, *args)
            k4 = df(f + h * k3, t + h, *args)
            t += h
            f += h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            f[f < non_negativity_cutoff] = non_negativity_cutoff
        sol[ti + 1] = f
    return sol


class survival_gen_func(object):
    """
    solves for the generating function of the copy number distribution of lineages
    in the traveling fitness wave. These generating functions are used to calculate
    lineage propagators conditional on not having sampled other branches.
    """

    def __init__(self, fitness_grid=None):
        """
        instantiates the class
        arguments:
            fitness_grid    discretization used for the solution of the ODEs
        """
        if fitness_grid is None:
            self.fitness_grid = np.linspace(-5, 8, 101)
        else:
            self.fitness_grid = fitness_grid

        self.L = len(self.fitness_grid)
        # precompute values necessary for the numerical evalutation of the ODE
        self.dx = self.fitness_grid[1] - self.fitness_grid[0]
        self.dxinv = 1.0 / self.dx
        self.dxsqinv = self.dxinv**2
        # dictionary to save interpolation objects of the numerical solutions
        self.phi_solutions = {}

    def integrate_phi(self, D, eps, T, save_sol=True, dt=None):
        """
        solve the equation for the generating function
        arguments:
        D   -- dimensionless diffusion constant. this is connected with the popsize
               via v = (24 D^2 log N)**(1/3)
        eps -- initial condition for the generating function
        T   -- grid of times on which the generating function is to be evaluated
        """
        phi0 = np.ones(self.L) * eps
        # if dt is not provided, use a heuristic that decreases with increasing diffusion constant
        if dt is None:
            dt = min(0.01, 0.001 / D)

        # set non-negative or very small values to non_negativity_cutoff
        sol = np.maximum(
            non_negativity_cutoff,
            integrate_rk4(self.dphi, phi0, T, dt, args=(D,)),
        )
        if save_sol:
            # produce and interpolation object to evaluate the solution at arbitrary time
            self.phi_solutions[(D, eps)] = interp1d(T, sol, axis=0)
        return sol

    def dphi(self, phi, t, D):
        """
        time derivative of the generating function
        """
        dp = np.zeros_like(phi)
        dp[1:-1] = (
            D * (phi[:-2] + phi[2:] - 2 * phi[1:-1]) * self.dxsqinv
            + (self.fitness_grid[1:-1]) * phi[1:-1]
            - phi[1:-1] ** 2
            - (phi[2:] - phi[:-2]) * 0.5 * self.dxinv
        )
        dp[0] = (
            0 * (phi[0] + phi[2] - 2 * phi[1]) * self.dxsqinv
            + (self.fitness_grid[0]) * phi[0]
            - phi[0] ** 2
            - (phi[1] - phi[0]) * self.dxinv
        )
        dp[-1] = (
            0 * (phi[-3] + phi[-1] - 2 * phi[-2]) * self.dxsqinv
            + (self.fitness_grid[-1]) * phi[-1]
            - phi[-1] ** 2
            - (phi[-1] - phi[-2]) * self.dxinv
        )
        return dp

    def integrate_prop_odeint(self, D, eps, x, t1, t2):
        """
        integrate the lineage propagator, accounting for non branching. THIS USES THE SCIPY ODE INTEGRATOR
        parameters:
        D   --  dimensionless diffusion constant
        eps --  initial condition for the generating function, corresponding to the sampling probability
        x   --  fitness at the "closer to the present" end of the branch
        t1  --  time closer to the present
        t2  --  times after which to evaluate the propagator, either a float or iterable of floats
        """
        if not np.iterable(
            t2
        ):  # if only one value is provided, produce a list with this value
            t2 = [t2]
        else:  # otherwise, cast to list. This is necessary to concatenate with with t1
            t2 = list(t2)

        if np.iterable(x):
            xdim = len(x)
        else:
            xdim = 1
            x = [x]

        # allocate array for solution: dimensions: #time points, #late fitness values, #fitness grid points
        sol = np.zeros((len(t2) + 1, len(x), self.L))
        # loop over late fitness values
        for ii, x_val in enumerate(x):
            # find index in fitness grid
            xi = np.argmin(x_val > self.fitness_grid)
            # init as delta function, normized
            prop0 = np.zeros(self.L)
            prop0[xi] = self.dxinv
            # propagate backwards and save in correct row
            sol[:, ii, :] = odeint(
                self.dprop_backward,
                prop0,
                [t1] + t2,
                args=((D, eps),),
                rtol=0.001,
                atol=1e-5,
                h0=1e-2,
                hmin=1e-4,
                printmessg=False,
            )

        return np.maximum(non_negativity_cutoff, sol)

    def integrate_prop(self, D, eps, x, t1, t2, dt=None):
        """
        integrate the lineage propagator, accounting for non branching. THIS USES THE SELF-MADE RK4.
        this is usually superior odeint, since we are integrating a system of ODEs with very different
        convergence criteria along the vector
        parameters:
        D   --  dimensionless diffusion constant
        eps --  initial condition for the generating function, corresponding to the sampling probability
        x   --  fitness at the "closer to the present" end of the branch
        t1  --  time closer to the present
        t2  --  times after which to evaluate the propagator, either a float or iterable of floats
        """
        if not np.iterable(t2):
            t2 = [t2]
        else:
            t2 = list(t2)

        if np.iterable(x):
            xdim = len(x)
        else:
            xdim = 1
            x = [x]
        # allocate array for solution: dimensions: #time points, #late fitness values, #fitness grid points
        if dt is None:
            dt = min(0.05, 0.01 / D)

        sol = np.zeros((len(t2) + 1, self.L, len(x)))
        # loop over late fitness values
        prop0 = np.zeros((self.L, len(x)))
        for ii, x_val in enumerate(x):
            # find index in fitness grid
            xi = np.argmin(x_val > self.fitness_grid)
            # init as delta function, normalized
            prop0[xi, ii] = self.dxinv

        # solve the entire matrix simultaneously.
        sol[:, :, :] = integrate_rk4(
            self.dprop_backward, prop0, [t1] + t2, dt, args=((D, eps),)
        )
        return np.maximum(non_negativity_cutoff, sol.swapaxes(1, 2))

    def dprop_backward(self, prop, t, params):
        """
        time derivative of the propagator.
        arguments:
            prop    value of the propagator
            t       time to evaluate the generating function
            params  parameters used to calculate the generating function (D,eps)
        """
        dp = np.zeros_like(prop)
        D = params[0]
        if (
            params in self.phi_solutions
        ):  # evaluate the generating function at the this time
            if t > self.phi_solutions[params].x[-1]:
                print(
                    (
                        "t larger than interpolation range! t=",
                        t,
                        ">",
                        self.phi_solutions[params].x[-1],
                    )
                )
            # evaluate at t if 1e-6<t<T[-2], boundaries otherwise
            tmp_phi = self.phi_solutions[params](
                min(max(1e-6, t), self.phi_solutions[params].x[-2])
            )
            # if propagator is 2 dimensional, repeat the generating function along the missing axis
            if len(prop.shape) == 2:
                tmp_phi = tmp_phi.repeat(prop.shape[1]).reshape(
                    [-1, prop.shape[1]]
                )
                fitness_grid = self.fitness_grid.repeat(prop.shape[1]).reshape(
                    [-1, prop.shape[1]]
                )
            else:
                fitness_grid = self.fitness_grid
        else:
            print(
                "solve_survival.dprop_backwards: parameters not in phi_solutions"
            )
            return None
        dp[1:-1] = (
            D * (prop[:-2] + prop[2:] - 2 * prop[1:-1]) * self.dxsqinv
            + (fitness_grid[1:-1] - 2 * tmp_phi[1:-1]) * prop[1:-1]
            - (prop[2:] - prop[:-2]) * 0.5 * self.dxinv
        )
        dp[0] = (
            0 * (prop[0] + prop[2] - 2 * prop[1]) * self.dxsqinv
            + (fitness_grid[0] - 2 * tmp_phi[0]) * prop[0]
            - (prop[1] - prop[0]) * self.dxinv
        )
        dp[-1] = (
            0 * (prop[-3] + prop[-1] - 2 * prop[-2]) * self.dxsqinv
            + (fitness_grid[-1] - 2 * tmp_phi[-1]) * prop[-1]
            - (prop[-1] - prop[-2]) * self.dxinv
        )
        return dp


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm

    mysol = survival_gen_func(np.linspace(-4, 8, 61))

    # calculate the generating function for different values of D and eps
    # the result is saved in mysol
    t_initial = 0.0
    for D in [0.2, 0.5]:
        plt.figure()
        plt.title("D=" + str(D))
        for eps, ls in zip([0.0001, 0.01, 1.0], ["-", "--", "-."]):
            sol = mysol.integrate_phi(D, eps, np.linspace(0, 20, 40))
            plt.plot(
                mysol.fitness_grid, mysol.phi_solutions[(D, eps)].y.T, ls=ls
            )

        plt.yscale("log")
        plt.xlabel("fitness")

    # use the precomputed generating functions to calculate the lineage propagators
    import time

    tstart = time.time()
    x_array = mysol.fitness_grid[5:-5]
    t_array = np.linspace(0.01, 7, 18)
    eps = 0.0001
    for D in [0.2]:  # , 0.5, 1.0]:
        sol_bwd = mysol.integrate_prop(
            D, eps, x_array, t_initial, t_initial + t_array
        )

        plt.figure()
        plt.suptitle("propagator D=" + str(D))
        for ti, t in enumerate(t_array[::2]):
            plt.subplot(3, 3, ti + 1)
            plt.title("t=" + str(np.round(t, 2)))
            plt.imshow(np.log(sol_bwd[2 * ti, :, :].T), interpolation="nearest")

        # plot mean offspring fitness as a function of time for difference ancestor fitness
        plt.figure()
        plt.title("Child fitness D=" + str(D) + " omega=" + str(eps))
        for yi in range(10, 50, 5):
            child_means = np.array(
                [
                    np.sum(sol_bwd[i, :, yi] * x_array)
                    for i in range(sol_bwd.shape[0])
                ]
            )
            child_means /= np.sum(sol_bwd[:, :, yi], axis=1)
            child_std = np.array(
                [
                    np.sum(sol_bwd[i, :, yi] * (x_array - child_means[i]) ** 2)
                    for i in range(sol_bwd.shape[0])
                ]
            )
            child_std /= np.sum(sol_bwd[:, :, yi], axis=1)
            child_std = np.sqrt(child_std)
            plt.plot(
                t_array, child_means[1:], ls="-", c=cm.jet(yi / 50.0), lw=2
            )
            plt.plot(t_array, child_std[1:], ls="--", c=cm.jet(yi / 50.0), lw=2)
        plt.xlabel("time to ancestor, child at t=" + str(t_initial))
        plt.ylabel("child mean fitness/std")

        # plot mean offspring fitness as a function of time for difference ancestor fitness
        plt.figure()
        plt.title("Ancestor fitness D=" + str(D) + " omega=" + str(eps))
        tmp_gauss = np.exp(-0 * mysol.fitness_grid**2 / 2)
        for xi in range(10, 45, 5):
            plt.plot(
                t_array,
                [
                    np.sum(sol_bwd[i, xi, :] * mysol.fitness_grid * tmp_gauss)
                    / np.sum(sol_bwd[i, xi, :] * tmp_gauss)
                    for i in range(1, sol_bwd.shape[0])
                ],
            )
        plt.xlabel("time to ancestor, child at t=" + str(t_initial))
        plt.ylabel("ancestor fitness")

    print((np.round(time.time() - tstart, 3), "seconds"))
