# src/gamecore/groebner.py
"""
Implements the algebraic geometry approach of Possieri & Sassano 2015 to compute all stabilizing
linear feedback Nash equilibria of an LQ game, by formulating the coupled algebraic Riccati equations
as a system of polynomial equations and using Groebner basis methods to solve them.

Adaptation for Speed-Up:
- We exclude the stability constraints from the Groebner basis computation, instead checking
  stability of each candidate solution afterwards. This significantly speeds up the Groebner step
  and furthermore avoids duplicate solutions introduced by the (quadratic) positivity constraints.
"""

import numpy as np
import sympy as sp
from scipy.optimize import root as scipy_root

from .strategy.linear_strategy import LinearStrategy
from .game.lq_game import LQGame

# --- Helpers ---------------------------------------------------------------

def _sympy_rational_matrix(A: np.ndarray):
    """Convert numpy array to sympy Matrix with Rational entries for stability."""
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    return sp.Matrix([[sp.Rational(str(float(A[i, j]))) for j in range(n)] for i in range(m)])

def _sym_idx_pairs(n: int):
    """Return list of index pairs (i,j) with i<=j for symmetric matrix entries.
       And a map from (i,j) -> variable index."""
    pairs = []
    idx_map = {}
    cnt = 0
    for i in range(n):
        for j in range(i, n):
            pairs.append((i, j))
            idx_map[(i, j)] = cnt
            idx_map[(j, i)] = cnt
            cnt += 1
    return pairs, idx_map

def _construct_X_symbols(n: int, prefix: str):
    """Return list of sympy symbols for the independent entries of symmetric X."""
    pairs, idx_map = _sym_idx_pairs(n)
    syms = sp.symbols(' '.join(f"{prefix}_{i}_{j}" for (i,j) in pairs))
    if type(syms) is not tuple:
        syms = (syms,)
    return list(syms), pairs, idx_map

def _reconstruct_symmetric_from_syms(syms, pairs, n):
    X = sp.zeros(n)
    for sym, (i,j) in zip(syms, pairs):
        X[i,j] = sym
        X[j,i] = sym
    return X

def is_hurwitz(A, tol_eig=-1e-8):
    eig = np.linalg.eigvals(A)
    return np.all(np.real(eig) < tol_eig)

def _compute_groebner_worker(args):
    polynomials_aug, vars_tuple, order = args
    import sympy as sp
    G = sp.groebner(polynomials_aug, *vars_tuple, order=order)
    return G

def build_block_R(block_dict, ms):
    blocks = []
    N = len(ms)
    for i in range(N):
        row = []
        for j in range(N):
            if (i, j) in block_dict:
                row.append(_sympy_rational_matrix(block_dict[(i, j)]))
            else:
                row.append(sp.zeros(ms[i], ms[j]))
        blocks.append(row)
    return sp.Matrix(blocks)

# --- Main function --------------------------------------------------------

def groebner_feedback_nash_equilibria(
    lq_game: LQGame,
    seed: int=0,
    numeric_tol: float = 1e-8,
    verbose: bool = False,
    algebraic_stability: bool = False,
) -> list[list[LinearStrategy]] | None:
    """
    Attempt to compute all stabilizing linear feedback Nash equilibria using 
    Possieri & Sassano 2015 "An algebraic geometry approach for the computation 
    of all linear feedback Nash equilibria in LQ differential games" approach.
    Compute the Groebner basis of the polynomial system derived from the coupled AREs,
    optionally including stability constraints algebraically (more expensive, duplicates),
    or checking stability post-hoc (faster, no duplicates).

    Parameters:
        ----------
    lq_game: LQGame
        The LQ game object containing system and player cost data.
    seed: int
        Random seed for generating separating polynomial.
    numeric_tol: float
        Tolerance for numerical root finding.
    verbose: bool
        Whether to print progress and result information.
    algebraic_stability: bool
        If True, include the stability constraints in the Groebner basis computation.
        This is more computationally expensive but guarantees that all returned solutions
        are stabilizing. It also introduces duplicate solutions.
        If False, stability is checked post-hoc for each candidate solution (recommended).

    Returns:
    ----------
    nash_strategies: nested list or None
        A list of lists of LinearStrategy objects corresponding to all feedback Nash equilibria. 
        Outer list over different equilibria, inner list over players. 
        Returns None if no stabilizing equilibria found.

    Notes:
    ----------
      - This is likely only tractable for small n (<=2) and small N (<=3).
    """
    # 0) Quick sanity checks & extract data
    try:
        sys = lq_game.system
        players = lq_game.players
        N = len(players)
        A = np.asarray(sys.A, dtype=float)
        n = A.shape[0]
        # extract Bi and Rij and Qi
        Bs = [np.asarray(B, dtype=float) for B in sys.Bs]  # sys.Bs is list of arrays
        # costs
        Qs = []
        Rs = []
        for p in players:
            Qs.append(np.asarray(p.cost.Q, dtype=float))
            # p.cost.R is dict[int, np.ndarray]
            Rs.append(p.cost.R)
    except Exception as e:
        raise ValueError("Error extracting LQ game data. Ensure lq_game is properly initialized.") from e
    if lq_game.type == "dynamic":
        raise ValueError("Groebner basis method only works for differential games.")

    # 1) Build symbolic unknowns: for X1..XN and X_{N+1} (Lyap)
    X_syms_all = []
    X_pairs_all = []
    X_idx_maps = []
    loop_range = range(N+1) if algebraic_stability else range(N)
    for i in loop_range:
        syms, pairs, idx_map = _construct_X_symbols(n, prefix=f"X{i+1}")
        X_syms_all.append(syms)
        X_pairs_all.append(pairs)
        X_idx_maps.append(idx_map)

    # 2) Build symbolic matrix representations
    X_mats = [_reconstruct_symmetric_from_syms(X_syms_all[i], X_pairs_all[i], n) for i in loop_range]

    # 3) Build polynomial equations from coupled ARE 

    # Convert parameters to sympy rational matrices for exactness
    A_sym = _sympy_rational_matrix(A)
    B_sym_list = [_sympy_rational_matrix(B) for B in Bs]
    Q_sym_list = [_sympy_rational_matrix(Q) for Q in Qs]
    R_sym_list = [build_block_R(Rs[i], lq_game.ms) for i in range(N)] 
    R_sym = build_block_R({(i,j): Rs[i][(i,j)] for i in range(N) for j in range(N) if (i,j) in Rs[i]}, lq_game.ms)
    # B = [B1 ... BN]
    B_stack = sp.Matrix.hstack(*B_sym_list)
    # BX = [B1^T X1; ...; BN^T XN]
    BX_stack = sp.Matrix.vstack(*[
        B_sym_list[i].T * X_mats[i] for i in range(N)
    ])

    # For each player i, form the matrix polynomial equation F^T X_i + X_i F + M_i = 0
    polynomials = []
    R_sym_inv = R_sym.inv()
    F = A_sym - B_stack * R_sym_inv * BX_stack
    for i in range(N):
        # compute M_i
        M_i = (
            Q_sym_list[i]
            + BX_stack.T * R_sym_inv.T * R_sym_list[i] * R_sym_inv * BX_stack
        )
        # form the matrix equation: F^T X_i + X_i F + M_i = 0
        left = F.T * X_mats[i] + X_mats[i] * F + M_i
        # left is a symbolic symmetric matrix; produce scalar polys for upper triangle entries
        for r in range(n):
            for c in range(r, n):
                polynomials.append(sp.simplify(left[r, c]))

    if algebraic_stability:
        # 4) Add stabilization constraints using X_{N+1} and Lyapunov equation as polynomial equations:
        # He(X_{N+1}(A - sum S_i X_i)) = -2 I -> He(...) + 2 I = 0
        A_cl = F
        X_lyap = X_mats[-1]
        He = (X_lyap * A_cl + (X_lyap * A_cl).T)  # symmetric part (scaling from paper not necessary, as X_lyap is free anyway)
        for r in range(n):
            for c in range(r, n):
                polynomials.append(sp.simplify(He[r, c] + (2 if r == c else 0)))

        # 5) Add positivity-of-leading-minors constraints via w_k: w_k^2 * M_k(X_{N+1}) - 1 = 0
        # Leading minors: compute symbolic determinant of top-left k x k submatrix
        w_syms = []
        for k in range(1, n+1):
            w = sp.symbols(f"w_{k}")
            w_syms.append(w)
            sub = X_lyap[:k, :k]
            Mk = sp.simplify(sp.Matrix(sub).det())
            polynomials.append(sp.simplify(w**2 * Mk - 1))

    # 6) Now collect all polynomial variables in a list & choose a random separating polynomial s(x)
    # Variables: all X_syms_all flattened + w_syms
    all_vars = []
    for syms in X_syms_all:
        all_vars.extend(syms)
    if algebraic_stability:
        all_vars.extend(w_syms)
    # Deduplicate sympy symbols (they should be unique already)
    # Choose random integer coefficients for separating polynomial s(x) = sum c_i * var_i
    rng = np.random.default_rng(seed)
    coeffs = [int(c) for c in rng.integers(1, 10, size=len(all_vars))]
    s_poly = sum(sp.Integer(c) * v for c, v in zip(coeffs, all_vars))

    # 7) Form augmented ideal {polynomials, sigma - s_poly}
    sigma = sp.symbols("sigma")
    polynomials_aug = polynomials + [sp.simplify(sigma - s_poly)]

    # 8) Compute Groebner basis with lex ordering, var order: put original vars first then sigma last (so sigma will be eliminated to univariate)
    # Variable ordering: we want lex with all_vars (in consistent order) > sigma, but sympy's groebner expects tuple of symbols ordering
    lex_order_vars = tuple(all_vars + [sigma])

    # try/exception logic adapted to typical sympy problems
    try:
        if verbose:
            print(f"Setup complete with {len(polynomials)} polynomials in {len(all_vars)} variables. "
                  f"Computing Groebner basis...")

        # Run Groebner in a separate process
        G = _compute_groebner_worker((polynomials_aug, lex_order_vars, 'lex'))

    except MemoryError as e:
        if verbose:
            print("Groebner computation ran out of memory.")
        return None


    # 9) If G computed, find polynomial in sigma (univariate). Extract basis polynomials that contain only sigma.
    solutions = []
    gb_list = list(G)
    sigma_polys = [g for g in gb_list if set(g.free_symbols) <= {sigma}]
    if len(sigma_polys) == 0:
        if verbose:
            print("No univariate sigma polynomial found in Groebner basis. Should not have happened. Aborting.")
        return None
    # pick polynomial of minimal degree in sigma (likely the desired eta)
    eta = min(sigma_polys, key=lambda p: sp.degree(sp.Poly(p, sigma)))
    # convert to numeric polynomial coefficients and find roots
    eta_poly = sp.Poly(sp.expand(eta), sigma)
    coeffs_eta = [complex(sp.N(c)) for c in eta_poly.all_coeffs()]
    # use numpy roots to find all (complex) sigma roots
    coeffs_real = np.array(coeffs_eta, dtype=complex)
    roots = np.roots(coeffs_real)
    if verbose:
        print(f"Univariate polynomial in sigma has degree {len(coeffs_real)-1} and {len(roots)} complex roots.")
    if len(roots) == 0:
        if verbose:
            print("No roots found for sigma polynomial. No solutions.")
        return None
    #  Filter roots: keep only real roots (imag part < tol)
    real_sigma = [float(np.real(r)) for r in roots if abs(np.imag(r)) < 1e-8]
    if len(real_sigma) == 0:
        if verbose:
            print("No real roots found for sigma polynomial. No solutions.")
        return None
    if verbose:
        if algebraic_stability:
            print(f"Found {len(real_sigma)} real roots for sigma. Contains duplicate solutions due to quadratic stability constraints (by factor 2^n).")
        else:
            print(f"Found {len(real_sigma)} real roots for sigma. Possibly contains non-stabilizing solutions, which will be filtered afterwards.")
    # For each real sigma, back-substitute using Groebner basis polynomials to recover xi's
    # The Groebner basis typically contains polynomials of the form xi - poly_i(sigma)
    # We'll attempt to solve linear substitution sequence: find polynomials where leading term is a symbol and rest depends on sigma
    for s_idx, s_val in enumerate(real_sigma):
        if verbose:
            print(f"Processing candidate {s_idx+1}/{len(real_sigma)} sigma = {s_val}...")
        subs = {sigma: sp.Rational(str(s_val))}
        candidate = {}
        ok = True
        # Try to parse gi forms: search for polynomials linear in one Xi with form Xi - poly_i(sigma)
        for sym in all_vars:
            # find a polynomial in GB that is linear and solves for sym
            found = False
            for g in gb_list:
                # check if g is of form sym + f(sigma, other constants) or sym - f(...)
                # simple heuristic: if g contains sym and degree in sym is 1 and no other symbol besides sigma
                free_syms = set(g.free_symbols)
                if sym in free_syms and free_syms - {sym, sigma} == set():
                    # try to isolate sym
                    try:
                        sol_for_sym = sp.solve(sp.Eq(g,0), sym)
                    except Exception:
                        sol_for_sym = []
                    if sol_for_sym:
                        val = sol_for_sym[0].subs(subs)
                        candidate[sym] = sp.N(val)
                        found = True
                        break
            if not found:
                # some vars may not be solvable directly from GB; fallback: numeric solve of full system with sigma fixed
                ok = False
                if verbose:
                    print(f"Could not isolate variable {sym} from Groebner basis; will fallback to numeric solve for sigma={s_val}.")
                break
        if not ok:
            # numeric fallback: solve the full polynomial system numerically with sigma fixed
            # build numeric functions and use scipy.root
            try:
                # assemble python functions for polynomials with sigma substituted
                poly_fun = sp.lambdify(all_vars, [p.subs({sigma: sp.Rational(str(s_val))}) for p in polynomials], 'numpy')
                # initial guess zeros
                x0 = np.zeros(len(all_vars))
                sol = scipy_root(lambda z: np.array(poly_fun(*z), dtype=float), x0, method='hybr', tol=numeric_tol)
                if not sol.success:
                    continue
                sol_vec = sol.x
                # map sol_vec back to matrices
                # build X_i numeric
                # convert to floats -> then check positivity and stability
                idx = 0
                X_numeric = []
                for i in loop_range:
                    syms_i = X_syms_all[i]
                    pairs = X_pairs_all[i]
                    Xi = np.zeros((n,n))
                    for k in range(len(syms_i)):
                        val = float(sol_vec[idx])
                        i0,j0 = pairs[k]
                        Xi[i0,j0] = val
                        Xi[j0,i0] = val
                        idx += 1
                    X_numeric.append(Xi)
            except Exception as e:
                if verbose:
                    print(f"Numeric solve failed for sigma={s_val}: {e}. Skipping...")
        else:
            # candidate dictionary contains sym->value for all variables
            # build numeric matrices
            X_numeric = []
            idx = 0
            for i in loop_range:
                pairs = X_pairs_all[i]
                Xi = np.zeros((n,n))
                for k in range(len(X_syms_all[i])):
                    sym = X_syms_all[i][k]
                    val = float(candidate[sym])
                    i0, j0 = pairs[k]
                    Xi[i0,j0] = val
                    Xi[j0,i0] = val
                X_numeric.append(np.asarray(Xi, dtype=float))

        ### compute K_i
        offsets = np.cumsum([0] + [m_i for m_i in lq_game.ms])
        # big shared R matrix
        blocks = []
        for i in range(lq_game.N):
            row = []
            for j in range(lq_game.N):
                row.append(lq_game.players[i].cost.R.get((i, j), np.zeros((lq_game.ms[i], lq_game.ms[j]))))
            blocks.append(row)
        R_big = np.block(blocks)
        # solve LGE for K_i
        rhs = np.vstack([Bs[i].T @ X_numeric[i] for i in range(lq_game.N)])
        K_vstack = np.linalg.solve(R_big, rhs)
        K_list = [K_vstack[offsets[i]:offsets[i+1]] for i in range(lq_game.N)]

        # check closed-loop stability
        Acl_num = A.copy()
        for i in range(N):
            Acl_num -= Bs[i] @ K_list[i]
        stable = is_hurwitz(Acl_num)
        # check X_{N+1} positive definite via eigenvalues
        if algebraic_stability:
            X_ly_num = X_numeric[-1]
            try:
                pd = np.all(np.linalg.eigvalsh(X_ly_num) > -1e-8)
            except np.linalg.LinAlgError as e:
                if verbose:
                    print(f"Eigenvalue computation failed for X_lyap at sigma={s_val}: {e}. Skipping...")
                pd = False
        else:
            pd = True  # no Lyap matrix in this mode
        if stable and pd:
            nash_strategies = [LinearStrategy(K=K_list[i]) for i in range(N)]
            solutions.append(nash_strategies)
            if verbose:
                print(f"Found stabilizing equilibrium for sigma={s_val}:")
                for i, K in enumerate(K_list):
                    print(f"  Player {i+1} K = {K}")
        else:
            if verbose:
                if algebraic_stability:
                    print(f"Candidate for sigma={s_val} not stabilizing or X_lyap not PD. Skipping...")
                else:
                    print(f"Candidate for sigma={s_val} not stabilizing. Skipping...")

    if verbose:
        print(f"Groebner computed; found {len(solutions)} stabilizing solution(s).")

    return solutions