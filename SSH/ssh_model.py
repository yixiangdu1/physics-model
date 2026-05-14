"""
SSH Model — Su-Schrieffer-Heeger 1D topological insulator.

Tight-binding chain with alternating hopping amplitudes t1 (intra-cell)
and t2 (inter-cell). The model hosts zero-energy edge states and a
topological phase transition when |t1| = |t2|.

Hamiltonian:
  H = Σ_n (t1 a_n† b_n + t2 a_{n+1}† b_n + h.c.)

Topological invariant: winding number ν
  ν = 1 (|t1| < |t2|) — nontrivial, edge states present
  ν = 0 (|t1| > |t2|) — trivial, no edge states
"""

import numpy as np
import matplotlib.pyplot as plt


# --- Hamiltonian construction ---

def build_ssh_open(N, t1, t2):
    """
    SSH Hamiltonian with open boundary conditions.
    Basis: |1,A>, |1,B>, |2,A>, |2,B>, ..., |N,A>, |N,B>
    Returns a 2N × 2N real symmetric matrix.
    """
    dim = 2 * N
    H = np.zeros((dim, dim))

    for n in range(N):
        a = 2 * n        # sublattice A in unit cell n
        b = 2 * n + 1    # sublattice B in unit cell n

        # Intra-cell hopping t1: a_n <-> b_n
        H[a, b] = t1
        H[b, a] = t1

        # Inter-cell hopping t2: b_n <-> a_{n+1}
        if n < N - 1:
            a_next = 2 * (n + 1)
            H[b, a_next] = t2
            H[a_next, b] = t2

    return H


def build_ssh_periodic(N, t1, t2):
    """
    SSH Hamiltonian with periodic boundary conditions.
    Includes the inter-cell hopping across the boundary: b_N <-> a_1.
    """
    H = build_ssh_open(N, t1, t2)

    # Wrap-around: last B connects to first A
    b_last = 2 * N - 1
    a_first = 0
    H[b_last, a_first] = t2
    H[a_first, b_last] = t2

    return H


def bloch_hamiltonian(k, t1, t2):
    """
    Bloch Hamiltonian h(k) in the sublattice (A,B) basis.
    h(k) = d·σ with d = (t1 + t2 cos k, t2 sin k, 0).

    Returns the 2×2 matrix at momentum k.
    """
    dx = t1 + t2 * np.cos(k)
    dy = t2 * np.sin(k)
    return np.array([[0, dx - 1j*dy],
                     [dx + 1j*dy, 0]], dtype=complex)


def band_structure(t1, t2, nk=200):
    """Dispersion relation E±(k) = ±|d(k)| for periodic BC."""
    k_vals = np.linspace(-np.pi, np.pi, nk)
    bands = np.empty((nk, 2))
    for i, k in enumerate(k_vals):
        dx = t1 + t2 * np.cos(k)
        dy = t2 * np.sin(k)
        eps = np.sqrt(dx**2 + dy**2)
        bands[i] = [-eps, eps]
    return k_vals, bands


# --- Topological invariants ---

def winding_number(t1, t2, nk=500):
    """
    Compute the winding number of d(k) around the origin.
    ν = (1/2π) ∮ dk (d̂ × ∂_k d̂)_z
    The SSH model has chiral symmetry → ν ∈ Z.
    """
    k_vals = np.linspace(-np.pi, np.pi, nk)
    dk = k_vals[1] - k_vals[0]

    dx = t1 + t2 * np.cos(k_vals)
    dy = t2 * np.sin(k_vals)
    norm = np.sqrt(dx**2 + dy**2)

    dx_hat = dx / norm
    dy_hat = dy / norm

    # ∂_k d̂
    ddx = np.gradient(dx_hat, dk)
    ddy = np.gradient(dy_hat, dk)

    integrand = dx_hat * ddy - dy_hat * ddx
    nu = np.trapezoid(integrand, k_vals) / (2 * np.pi)

    return round(nu.real)


def zak_phase(t1, t2, nk=500):
    """
    Zak phase = π × winding number (mod 2π) for the SSH model.
    Numerically computed from the lower-band Berry connection.
    """
    k_vals = np.linspace(-np.pi, np.pi, nk)
    dk = k_vals[1] - k_vals[0]

    zak = 0.0
    prev = None
    for k in k_vals:
        dx = t1 + t2 * np.cos(k)
        dy = t2 * np.sin(k)
        eps = np.sqrt(dx**2 + dy**2)
        # Lower band eigenvector
        if eps > 1e-12:
            psi = np.array([dx - 1j*dy, -eps]) / (np.sqrt(2) * eps)
        else:
            psi = np.array([0, 1.0])  # gap closing point
        if prev is not None:
            overlap = np.vdot(prev, psi)
            zak += np.angle(overlap)  # discrete Berry connection
        prev = psi

    return zak % (2 * np.pi)


# --- Plotting ---

def plot_spectrum_and_edge(t1, t2, N=20, ax=None):
    """Eigenvalues vs state index for open BC, highlighting edge states."""
    H_open = build_ssh_open(N, t1, t2)
    evals, evecs = np.linalg.eigh(H_open)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    colors = []
    for i, (e, vec) in enumerate(zip(evals, evecs.T)):
        prob = np.abs(vec)**2
        weight_left = np.sum(prob[:4])     # left edge (first 2 cells)
        weight_right = np.sum(prob[-4:])   # right edge (last 2 cells)
        if abs(e) < 1e-6 and (weight_left + weight_right) > 0.5:
            colors.append("red")
        else:
            colors.append("gray")

    ax.scatter(range(2*N), evals, c=colors, s=40, zorder=5,
               edgecolors="black", linewidths=0.5)
    ax.axhline(0, color="red", ls="--", alpha=0.3)
    ax.set_xlabel("State index")
    ax.set_ylabel("Energy")
    ax.set_title(f"Open BC — t1={t1:.1f}, t2={t2:.1f}")
    return ax


def plot_wavefunction(t1, t2, N=20, ax=None):
    """Probability density of the zero-energy edge states."""
    H_open = build_ssh_open(N, t1, t2)
    evals, evecs = np.linalg.eigh(H_open)

    # Find near-zero modes
    tol = 1e-6
    edge_modes = evecs.T[np.abs(evals) < tol]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    for i, vec in enumerate(edge_modes):
        prob = np.abs(vec)**2
        label = "Edge mode" if i == 0 else None
        ax.plot(range(2*N), prob, 'o-', ms=4, lw=1.2, label=label)

    if len(edge_modes) == 0:
        ax.text(0.5, 0.5, "No edge states (trivial phase)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="gray")

    ax.set_xlabel("Site index")
    ax.set_ylabel(r"$|\psi|^2$")
    ax.set_title(f"Edge state wavefunction — t1={t1:.1f}, t2={t2:.1f}")
    if len(edge_modes) > 0:
        ax.legend()
    return ax


def main():
    np.random.seed(42)
    N = 30           # number of unit cells

    # --- Phase diagram scan ---
    t2 = 1.0
    t1_vals = np.linspace(0.0, 3.0, 31)

    # --- Figure 1: Spectrum evolution + winding number ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    nu_vals = []
    gap_vals = []

    for t1 in t1_vals:
        H_open = build_ssh_open(N, t1, t2)
        evals = np.linalg.eigvalsh(H_open)

        E_sorted = np.sort(np.abs(evals))
        gap = E_sorted[1] if len(E_sorted) > 1 else 0  # gap = |E_1| after E_0
        gap_vals.append(gap)

        nu = winding_number(t1, t2)
        nu_vals.append(nu)

        # Paint by phase
        color = "#2196F3" if nu == 1 else "#FF9800"
        ax1.scatter([t1] * len(evals), evals, c=color, s=3, alpha=0.6)

    ax1.axhline(0, color="red", ls="--", alpha=0.3, lw=0.8)
    ax1.set_xlabel("t1 / t2")
    ax1.set_ylabel("Energy")
    ax1.set_title("Spectral flow (open BC)")

    ax2.plot(t1_vals, gap_vals, 'k-', lw=1.5)
    ax2.fill_between(t1_vals, 0, gap_vals,
                      where=np.array(nu_vals)==1, alpha=0.15, color="#2196F3",
                      label="Topological (ν=1)")
    ax2.fill_between(t1_vals, 0, gap_vals,
                      where=np.array(nu_vals)==0, alpha=0.15, color="#FF9800",
                      label="Trivial (ν=0)")
    ax2.axvline(t2, color="red", ls="--", alpha=0.5, label="t1 = t2")
    ax2.set_xlabel("t1 / t2")
    ax2.set_ylabel("Energy gap")
    ax2.set_title("Bulk gap + Winding number")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("ssh_phase_diagram.png", dpi=150)

    # --- Figure 2: Band structure for two phases ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, (t1, title) in zip(axes, [(0.5, "Topological (t1 < t2)"),
                                        (1.0, "Critical (t1 = t2)"),
                                        (1.5, "Trivial (t1 > t2)")]):
        k, bands = band_structure(t1, t2)
        ax.plot(k, bands[:, 0], '#2196F3', lw=1.5)
        ax.plot(k, bands[:, 1], '#2196F3', lw=1.5)
        ax.fill_between(k, bands[:, 0], bands[:, 1], alpha=0.1, color="#2196F3")
        ax.axhline(0, color="gray", ls="--", alpha=0.3)
        ax.set_xlabel("k")
        ax.set_ylabel("E")
        ax.set_title(title)
        nu = winding_number(t1, t2)
        zak = zak_phase(t1, t2)
        ax.text(0.95, 0.95, f"ν={nu}, γ={zak/np.pi:.1f}π",
                transform=ax.transAxes, va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    fig.tight_layout()
    fig.savefig("ssh_band_structure.png", dpi=150)

    # --- Figure 3: Edge states in topological vs trivial phase ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Topological: t1=0.5, t2=1.0
    plot_spectrum_and_edge(0.5, 1.0, N=N, ax=axes[0, 0])
    axes[0, 0].set_title("Spectrum — Topological (t1=0.5, t2=1.0)")
    plot_wavefunction(0.5, 1.0, N=N, ax=axes[0, 1])

    # Trivial: t1=1.5, t2=1.0
    plot_spectrum_and_edge(1.5, 1.0, N=N, ax=axes[1, 0])
    axes[1, 0].set_title("Spectrum — Trivial (t1=1.5, t2=1.0)")
    plot_wavefunction(1.5, 1.0, N=N, ax=axes[1, 1])

    fig.tight_layout()
    fig.savefig("ssh_edge_states.png", dpi=150)

    # --- Text summary ---
    print("SSH Model — Results Summary")
    print("=" * 50)
    print(f"Fixed t2 = {t2}")
    print()
    for t1 in [0.5, 1.0, 1.5]:
        nu = winding_number(t1, t2)
        zak = zak_phase(t1, t2)
        phase = "TOPOLOGICAL" if nu == 1 else "TRIVIAL"
        if t1 == 1.0:
            phase = "CRITICAL (gap closing)"
        print(f"  t1={t1:.1f}: ν={nu}, Zak phase={zak/np.pi:.2f}π — {phase}")

    H_open = build_ssh_open(N, 0.5, 1.0)
    evals, _ = np.linalg.eigh(H_open)
    n_edge = np.sum(np.abs(evals) < 1e-6)
    print(f"\nEdge states at t1=0.5: {n_edge} near-zero modes detected")
    print("Plots saved: ssh_phase_diagram.png, ssh_band_structure.png, ssh_edge_states.png")

    plt.show()


if __name__ == "__main__":
    main()
