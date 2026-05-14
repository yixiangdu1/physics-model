"""
1D Ising Model — Metropolis Monte Carlo simulation.
Hamiltonian: H = -J Σ s_i s_{i+1}   (J > 0, s_i = ±1)
"""

import numpy as np
import matplotlib.pyplot as plt


def init_lattice(N):
    """Random spin configuration of size N, each spin ±1."""
    return np.random.choice([-1, 1], size=N)


def total_energy(spins, J):
    """Compute energy E = -J Σ s_i s_{i+1} with periodic BC."""
    return -J * np.sum(spins * np.roll(spins, -1))


def total_magnetization(spins):
    """Magnetization M = Σ s_i."""
    return np.sum(spins)


def metropolis_step(spins, J, beta):
    """
    One sweep: attempt to flip each spin in random order.
    Accept with probability min(1, exp(-β ΔE)).
    """
    N = len(spins)
    order = np.random.permutation(N)
    for i in order:
        # ΔE = 2 J s_i (s_{i-1} + s_{i+1})  (periodic BC)
        left = spins[(i - 1) % N]
        right = spins[(i + 1) % N]
        dE = 2 * J * spins[i] * (left + right)
        if dE <= 0 or np.random.random() < np.exp(-beta * dE):
            spins[i] *= -1
    return spins


def simulate(N, J, T, n_thermal, n_measure, n_sample):
    """
    Run Monte Carlo at temperature T.
    Returns arrays of magnetization and energy per sample sweep.
    """
    spins = init_lattice(N)
    beta = 1.0 / T

    # Thermalize
    for _ in range(n_thermal):
        metropolis_step(spins, J, beta)

    # Measure
    mags = np.empty(n_sample)
    energies = np.empty(n_sample)
    for k in range(n_sample):
        for _ in range(n_measure):
            metropolis_step(spins, J, beta)
        mags[k] = total_magnetization(spins)
        energies[k] = total_energy(spins, J)

    return mags, energies


def exact_free_energy_per_spin(T, J=1.0):
    """Analytical free energy per spin for infinite 1D Ising."""
    if T == 0:
        return -J
    beta = 1.0 / T
    return -np.log(2 * np.cosh(beta * J)) / beta


def exact_energy_per_spin(T, J=1.0):
    """Analytical energy per spin: <E>/N = -J tanh(βJ)."""
    T = np.asarray(T, dtype=float)
    scalar = T.ndim == 0
    if scalar:
        T = np.array([T])
    beta = np.divide(1.0, T, where=T > 0, out=np.full_like(T, np.inf))
    result = -J * np.tanh(beta * J)
    result[T == 0] = -J
    return result.item() if scalar else result


def exact_heat_capacity_per_spin(T, J=1.0):
    """Analytical specific heat per spin: C/N = (βJ / cosh(βJ))^2."""
    T = np.asarray(T, dtype=float)
    scalar = T.ndim == 0
    if scalar:
        T = np.array([T])
    beta = np.divide(1.0, T, where=T > 0, out=np.full_like(T, np.inf))
    result = (beta * J / np.cosh(beta * J)) ** 2
    result[T == 0] = 0.0
    return result.item() if scalar else result


def main():
    np.random.seed(42)

    N = 200              # chain length
    J = 1.0              # coupling
    n_thermal = 2000     # thermalization sweeps
    n_measure = 10       # sweeps between measurements
    n_sample = 500       # number of measurements

    T_vals = np.linspace(0.1, 5.0, 30)

    energy_per_spin = []
    energy_err = []
    mag_abs = []
    mag_err = []
    heat_cap = []
    heat_cap_err = []

    for T in T_vals:
        mags, energies = simulate(N, J, T, n_thermal, n_measure, n_sample)

        E = energies / N
        energy_per_spin.append(np.mean(E))
        energy_err.append(np.std(E) / np.sqrt(n_sample))

        M_abs = np.abs(mags) / N
        mag_abs.append(np.mean(M_abs))
        mag_err.append(np.std(M_abs) / np.sqrt(n_sample))

        beta = 1.0 / T
        C = beta**2 * (np.mean(energies**2) - np.mean(energies)**2) / N
        heat_cap.append(C)
        heat_cap_err.append(beta**2 * np.std(energies**2 - energies.mean()**2) / np.sqrt(n_sample) / N)

    energy_per_spin = np.array(energy_per_spin)
    energy_err = np.array(energy_err)
    mag_abs = np.array(mag_abs)
    mag_err = np.array(mag_err)
    heat_cap = np.array(heat_cap)
    heat_cap_err = np.array(heat_cap_err)

    # --- Analytical ---
    T_fine = np.linspace(0.01, 5.0, 200)
    E_exact = exact_energy_per_spin(T_fine, J)
    C_exact = exact_heat_capacity_per_spin(T_fine, J)

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.errorbar(T_vals, energy_per_spin, yerr=energy_err, fmt='o', ms=4, capsize=2, label='MC')
    ax.plot(T_fine, E_exact, '-', label='Exact')
    ax.set_xlabel('T')
    ax.set_ylabel('<E> / N')
    ax.set_title('Energy per spin')
    ax.legend()

    ax = axes[1]
    ax.errorbar(T_vals, mag_abs, yerr=mag_err, fmt='o', ms=4, capsize=2)
    ax.set_xlabel('T')
    ax.set_ylabel('<|M|> / N')
    ax.set_title('Magnetization per spin')

    ax = axes[2]
    ax.errorbar(T_vals, heat_cap, yerr=heat_cap_err, fmt='o', ms=4, capsize=2, label='MC')
    ax.plot(T_fine, C_exact, '-', label='Exact')
    ax.set_xlabel('T')
    ax.set_ylabel('C / N')
    ax.set_title('Heat capacity per spin')
    ax.legend()

    plt.tight_layout()
    plt.savefig("ising1d_results.png", dpi=150)
    plt.show()
    print("Done — plot saved to ising1d_results.png")


if __name__ == "__main__":
    main()
