"""config.py — every design constant of the simulation, as specified in the manuscript
(v1.3: one-day profile, 2026-09-28).

Section numbers refer to the manuscript. Items marked ASSUMPTION are not fixed by the manuscript
and were chosen when the code was written; each is documented at its definition below and can be
changed here (and only here).

v1.3 keeps every factor level, every cell, every method and every performance measure of the
manuscript. What changed is *how* the work is scheduled and sampled, not *what* is estimated:
  * all cells with the same array shapes are fitted in one GPU call (cross-cell batching),
  * chains start at the prior median (with jitter) instead of uniform(-2, 2),
  * warm-up 400 (was 1,000); sampling iterations unchanged (1,000) because the R-hat < 1.01 rule
    on ~800 quantities needs that many draws to be free of chance exceedances,
  * rolling-origin refits warm-start from the previous origin (150 warm-up iterations),
  * the model-based baselines of Study B (M0, NB no-pooling, NB complete-pooling) are refitted every
    12 months and propagated by the same particle filter in between (B_BASELINE_STRIDE = 2),
  * replications are filled in passes: pass 1 = PASS1_REPS per cell everywhere, later passes top up
    to the manuscript's 100 / 200 (append-only results, resumable at any point).
"""
import math

DESIGN = "v1.4-oneday"

# ------------------------------------------------------------------ MCMC (§4.1 Computation and diagnostics)
CHAINS = 4
WARMUP = 400                      # v1.3: 400 (manuscript v1: 1,000) — chains start at the prior median
SAMPLES = 500                     # v1.3: 500 (manuscript v1: 1,000) — 4 x 500 = 2,000 draws give ESS > 400 and R-hat < 1.01 on all ~800 quantities in the reference cell
MAX_TREE_DEPTH = 10               # v1.3.1: 10 (was 8)
TARGET_ACCEPT = 0.95              # v1.4: 0.95 again. v1.3.1 had raised it to 0.99 because 46% of the Study B origin fits
                                  #   diverged at 0.95 (Mac run, 2026-09-29); with the (m_bar, d_bar) reparameterisation of
                                  #   models.ews_model the design fits are divergence-free at 0.95 (tests, 2026-09-30) and
                                  #   0.99 only doubled the leapfrog steps
REFIT = (1000, 0.95, 10, 1000)    # repeated once on the same data with 1,000 warm-up and 1,000 sampling iterations
                                  # (warmup, target acceptance, tree depth, samples) ...
REFIT_ACCEPT_DIV = 0.99           # ... at target acceptance 0.99 when the first fit had divergent transitions
WARM_START_WARMUP = 150           # rolling origins after the first: start at the previous origin's draws,
                                  # step size and mass matrix, re-adapt for this many iterations
RHAT_MAX = 1.01                   # rank-normalised R-hat < 1.01
ESS_MIN = 400                     # bulk and tail ESS > 400 on the reported quantities
DIV_MAX = 0                       # no divergent transitions
# R-hat: every parameter except the standard-normal variables of the non-centred parameterisations —
# the trend innovations, the item effects u_raw and the group dispersions a_raw — whose functionals
# (the trend paths m_t and delta_t, log C_i and alpha'_g) are checked instead, the horseshoe local
# scales and the latent fill-in values of missing indicators (auxiliary variables; v1.1 / v1.3.1)
RHAT_EXCLUDE = ("eta_raw", "zeta_raw", "s_eta", "s_zeta", "lam_hs", "x_mis", "u_raw", "a_raw",
                "z_hs", "nu_hs")                # z_hs, nu_hs: the horseshoe local scales of the A6 arm (models.py)
RHAT_EXTRA_DET = ("m_t", "delta_t", "logC", "alpha")
# reported quantities (§5.1 performance measures, Table 5a): alpha', C, beta, trend paths, w, kappa_w
ESS_SITES = ("beta", "alpha", "w", "kappa_w", "logC", "m_t", "delta_t")
# initial point of every chain in the unconstrained space (prior medians; m1 is set from the data),
# plus N(0, INIT_JITTER^2) jitter per chain. Sites not listed start at 0 (= prior median for the
# standard-normal, uniform and unit-scale sites).
INIT_UNCONSTRAINED = dict(kappa_w=math.log(0.674), tau_C=math.log(0.337), m_alpha=math.log(2.0),
                          tau_alpha=math.log(0.337), sigma_eta=math.log(0.03), tau_zeta=math.log(0.01),
                          alpha1=math.log(2.0), alpha_i=math.log(2.0), phi=math.log(16.8))
INIT_JITTER = 0.2
INIT_JITTER_SITE = dict(m_bar=0.05, d_bar=0.002)   # v1.4: per-site jitter where 0.2 is far outside the posterior
# exact reparameterisation of the innovation vectors: mean-level / mean-slope coordinate split off
# (see models.innov_basis). Same posterior; only the coordinates the sampler moves in.
INNOV_SPLIT = False
# v1.4 (CPU): fast log density (models.ews_model) and sampler options (nuts_gpu.py)
COUNT_TABLE = 64                  # counts below this enter through per-item tail tables, larger ones exactly
COUNT_OVERFLOW = 256              # slots for counts >= COUNT_TABLE per replication
CHAIN_MAP = "map"                 # "map": chains (and replications of a call) one after another inside the
                                  # compiled sampler (CPU: no lane waste); "vmap": vectorised (GPU)
CHUNK_DIR = None                  # v1.4: directory of segmented-fit checkpoints (set on the Mac VM, whose commands
CHUNK_ITERS = 50                  #   last at most 3 minutes); None = one compiled call per fit
DEADLINE_EPOCH = None             # epoch seconds: segmented fits checkpoint and raise nuts_gpu.TimeUp before it
BOOT_SECONDS_MAX = 100            # segmented runs: time the bootstrap calibration needs (else TimeUp before it)
ML_SECONDS_MAX = 40               # segmented runs: time one origin of the ML baselines needs
INIT_METRIC = True                # warm-up starts from 1 / (Hessian diagonal) for the low-dimensional sites
                                  # (numpyro's default start is the identity); adaptation runs as before
DENSE_BLOCKS = ()                 # e.g. (("m1", "s_eta"), ("d1", "s_zeta"), ("kappa_w", "e_w")) with INNOV_SPLIT

# ------------------------------------------------------------------ priors (eqs. 18-21)
PRIOR = dict(beta_sd=1.0, kappa_sd=1.0, tauC_sd=0.5, m_alpha_mean=math.log(2.0), m_alpha_sd=1.0,
             tau_alpha_sd=0.5, sigma_eta_sd=0.1, tau_zeta_sd=0.1, m1_sd=1.0, d1_sd=0.05,
             phi_shape=2.0, phi_rate=0.1, gamma_sd=1.0)
# §4.2: s_w = 0.1 + CR; with two sub-criteria the pairwise matrix is always consistent (CR = 0)
S_W = 0.1
# ASSUMPTION: the simulation's AHP weights point in the true direction, w_AHP = w_true / kappa_true
# with kappa_true = 0.8 (Table 5a: w = (0.5, 0.3), kappa_w = 0.8) -> w_AHP = (0.625, 0.375)
W_AHP = (0.625, 0.375)
# ASSUMPTION: no-pooling variant ("alpha'_i and u_i free", §4.1): alpha'_i ~ LogNormal(log 2, 1) per item,
# u_i ~ N(0, 1) centred within group (no shared scale tau_C)
U_NOPOOL_SD = 1.0
# ASSUMPTION: regularised horseshoe (§3.5, Piironen & Vehtari 2017) on the level innovations with
# global scale sigma_eta, local lambda ~ HalfCauchy(1), slab scale 1 on the log scale, slab df 4
HS_SLAB_SCALE, HS_SLAB_DF = 1.0, 4.0
LOGMU_CLIP = (-20.0, 10.0)

# ------------------------------------------------------------------ data-generating process (Table 4)
G = 3
K = 3
RHO_X = 0.8
BETA = (0.4, 0.3, 0.2)
W_TRUE = (0.5, 0.3)
KAPPA_TRUE = 0.8
SIGMA_U = 0.3
ALPHAS = ("0.5", "2", "10", "inf")
MUS = ("0.05", "0.2", "0.5")
REGIMES = ("stable", "gradual", "abrupt")
SIZES = ((10, 60), (30, 120), (30, 240))      # I in {10, 30}, T in {60, 120, 240}: three (I, T) levels
GRADUAL_SLOPE = 0.02                          # per month from t0 = T/2
ABRUPT_JUMP = math.log(3.0)
MISS_DEFAULT = 0.05
SEV_LOGNORMAL = (-2.0, 0.7)                   # log-normal(-2, 0.7^2) truncated to (0, 1)
SEV_PARETO = (1.5, 0.05)                      # Pareto shape 1.5, minimum 0.05, truncated at 1
REF = dict(alpha="2", mu="0.2", regime="gradual", I=30, T=120)

# ------------------------------------------------------------------ replications: manuscript targets and pass 1
A_REPS, A_REPS_REF = 100, 200                 # §5.1 (final targets, reached by top-up passes)
B_REPS, B_REPS_REF = 100, 200
R_REPS = 100
PASS1_REPS = dict(A=30, A_ref=60, B=30, B_ref=60, R=30)   # one-day pass: every cell, every method
# Monte Carlo standard errors at pass 1 (Morris et al. 2019): coverage of a 90% interval with 30
# replications has MCSE sqrt(0.9*0.1/30) = 0.055; Study B alarm metrics pool 30 items x ~80 months
# per replication, so 30 replications give cell means far below the between-cell differences.

# ------------------------------------------------------------------ Study A (§5.1, §5.2)
A_VARIANTS = ("hier", "none", "complete")     # compared in every cell
A_HORSESHOE_REGIMES = ("abrupt",)            # horseshoe variant evaluated only in the abrupt regime (§5.2)
# cell filters for reduced profiles (None = every cell): the pooling variants (none / complete) and
# the horseshoe variant are run only in cells whose factor levels are in these sets
A_POOLING_CELLS = None                        # e.g. dict(regime=("gradual",), alpha=("2",), size=((10, 60), (30, 120)))
A_HORSESHOE_CELLS = None                      # e.g. dict(alpha=("2",))
# per-cell replication overrides of the reduced profiles (None = the pass count PASS1_REPS['A'] / 'A_ref'):
# the cells of the pooling / horseshoe slices get at least A_SLICE_REPS (Fig. 6, the changepoint
# comparison), the cells of A_LONG_SIZE at most A_LONG_REPS (they enter only the pooled sample-size
# contrasts of §5.2 and cost about four fits of the reference size each)
A_SLICE_REPS = None
A_LONG_REPS = None
A_LONG_SIZE = (30, 240)

# ------------------------------------------------------------------ Study B (§5.3)
B_ALPHAS = ("0.5", "2")
B_MUS = ("0.05", "0.2")
ORIGINS = tuple(range(36, 115, 6))            # 36, 42, ..., 114 (14 origins)
DECISION_END = 117                            # decisions every month from 36 to 117
HORIZON = 3
PI_LEVELS = (0.048, 0.167, 0.5)               # cost ratios 1:20, 1:5, 1:1 (eq. (19) of the manuscript)
PERSIST = 2                                   # 2-of-2
RHO_SHIFT = 1.5                               # eq. (24)
BOOT_PATHS = 200                              # §5.3 (the §4.4 default of 500 is for applications)
ARL0_TARGET = 24
# ASSUMPTION: bootstrap paths are as long as the monitoring span (36 -> 117); thresholds searched on
# Q_GRID, persistence strengthened 2 -> 3 -> 4 if no threshold reaches the target (§4.4)
BOOT_MONTHS = DECISION_END - ORIGINS[0]
# ASSUMPTION: a stable-regime bootstrap path holds the drawn level fixed (as the stable regime of
# Table 4); set True to let it walk with the drawn sigma_eta instead
BOOT_LEVEL_WALK = False
BOOT_CHUNK = 32                               # replications per vectorised bootstrap call (memory)
BOOT_PATH_BLOCK = 25                          # v1.4 segmented runs: bootstrap paths per checkpointed block
Q_GRID = tuple([round(0.5 + 0.05 * k, 2) for k in range(10)] + [0.975, 0.99])
PERSIST_MAX = 4
B_MODELS = ("M2", "M0", "NBnone", "NBcomplete")   # proposal + model-based baselines, full pipeline each
# v1.3: the model-based baselines are refitted at every B_BASELINE_STRIDE-th origin (36, 48, ..., 108
# for stride 2) and propagated by the particle filter in between; M2 is refitted at every origin
# (B_M2_STRIDE = 1; a reduced profile may set 2 = every twelve months, §5.3 sentence to match).
B_BASELINE_STRIDE = 2
B_M2_STRIDE = 1
# sigma-rule on the plug-in SRI (§5.1 Baselines): 24-month window, limits at -1, -1.5, -2 sigma
SIGMA_WINDOW, SIGMA_LIMITS = 24, (1.0, 1.5, 2.0)
# ASSUMPTION: plug-in SRI = SRI at the posterior means of mu and alpha' of the M2 filter; the 24-month
# history before the first decision comes from the smoothed fit at the first origin
# ASSUMPTION: EWMA (lambda 0.2) and Poisson CUSUM (Lucas 1985, shift factor RHO_SHIFT) on item counts,
# in-control mean from months 13-36 (frozen), limits Poisson-calibrated to a nominal ARL0 of 24 months
EWMA_LAMBDA = 0.2
CHART_PHASE1 = (12, 36)                       # months 13..36 as 0-based columns 12..35
# ASSUMPTION: machine-learning baselines (§5.1): gradient boosting = XGBoost (Chen & Guestrin 2016);
# LSTM and TFT on 12-month sequences; hyper-parameters chosen on the last 12 labelled months of each
# origin's training range (inner validation), early stopping on the same split
ML_LOOKBACK = 12
ML_VAL_MONTHS = 12
XGB_GRID = ({"max_depth": 3, "learning_rate": 0.05}, {"max_depth": 3, "learning_rate": 0.1},
            {"max_depth": 6, "learning_rate": 0.05}, {"max_depth": 6, "learning_rate": 0.1})
XGB_WORKERS = None                            # None = all CPU cores (joblib over replications)
NN_HIDDEN = (16, 32)
NN_LR = (1e-3, 3e-3)
NN_STEPS, NN_EVAL_EVERY = 400, 20
NN_CHUNK = 24                                 # replications per vectorised training call (TFT needs ~1 GB per replication at late origins)
ML_BATCH = 256                                # v1.4: minibatch size of the LSTM / TFT Adam steps (0 = full batch)
# conformal sets for next-month counts (§4.3): APS score, ACI (Gibbs & Candes 2021)
CONF_ALPHA = 0.10
ACI_GAMMA = 0.005                             # ASSUMPTION: ACI step size
CONF_NMAX = 40
CONF_MIN_CALIB = 30                           # ASSUMPTION: scores needed before sets are formed
# severity and the severe level (§4.4)
# ASSUMPTION: severe additionally requires p_it(3) * E[S_it] >= SEVERE_LOSS; GPD fitted to exceedances
# over the 90th percentile of observed S (at least GPD_MIN_EXC exceedances, else body only)
SEVERE_LOSS = 0.10
SEV_CAP_FRAC = 0.4                            # v1.4: capacity of the gathered severity data (share of item-periods)
GPD_QUANTILE, GPD_MIN_EXC = 0.90, 10

# ------------------------------------------------------------------ robustness (§5.4)
R_ALARM_HAWKES = True                         # False: Hawkes cells get the recovery fit only (reduced profiles)
HAWKES_N = (0.0, 0.3, 0.6)
HAWKES_KD = (0.3, 1.0, 4.0)
MEAS_ERR = 0.5
MISS_HIGH = 0.15
COLD_FRAC = 0.20                              # observed only in the last T/4 periods
# ASSUMPTION: "indicator lag misspecified by one period": truth uses x_{t-2}, the model x_{t-1}
LAG_TRUE_MISSPEC = 2

# ------------------------------------------------------------------ scheduling (v1.3)
FAIL_WARN = 0.10                              # failure rate above which a cell is written to warnings.log
MAX_CHAINS = 6000                             # upper bound on replications x chains in one GPU call
DIAG_CHUNK = 32                               # replications per chunk for the on-device diagnostics / summaries (memory)


# ------------------------------------------------------------------ profiles
PROFILE = "oneday"


def apply_profile(name):
    """Switch the constants of a reduced profile. 'oneday' = the Colab A100 profile above (default).
    'mac_min' = the smallest run that still fills every result placeholder of §5.2-5.5, for a
    10-core MacBook: fewer replications, model-based baselines refitted every
    24 months, pooling and horseshoe variants only where Fig. 6 / §5.2 (iii) need them, alarm
    pipeline only for the non-Hawkes robustness arms, one network configuration.
    'colab_min' = the same scope on a GPU (chunk sizes of the oneday profile).
    't4_min' = the colab_min scope cut to what a Colab T4 can do in one or two sessions: 5
    replications per cell (Study B reference cell 10, Study A reference cell 20), the two nine-cell
    Study A slices at 10 (pooling and horseshoe variants at (10, 60) and (30, 120) only — the mu x I
    slice of Fig. 6), the (30, 240) cells at 3 (hierarchical only)."""
    global PROFILE, DESIGN, PASS1_REPS, B_BASELINE_STRIDE, A_POOLING_CELLS, A_HORSESHOE_CELLS, R_ALARM_HAWKES
    global NN_HIDDEN, NN_LR, NN_STEPS, XGB_GRID, XGB_WORKERS, DIAG_CHUNK, NN_CHUNK, MAX_CHAINS
    global A_SLICE_REPS, A_LONG_REPS, B_M2_STRIDE
    PROFILE = name
    if name == "oneday":
        return
    if name == "mac_min":
        DESIGN = "v1.4-mac_min"
        PASS1_REPS = dict(A=10, A_ref=20, B=10, B_ref=20, R=10)
        B_BASELINE_STRIDE = 4                                    # baselines at origins 36, 60, 84, 108
        A_POOLING_CELLS = dict(regime=("gradual",), alpha=("2",))
        A_HORSESHOE_CELLS = dict(alpha=("2",))
        R_ALARM_HAWKES = False
        NN_HIDDEN, NN_LR, NN_STEPS = (16,), (3e-3,), 200
        XGB_GRID = ({"max_depth": 3, "learning_rate": 0.1}, {"max_depth": 6, "learning_rate": 0.1})
        XGB_WORKERS = 1
        DIAG_CHUNK, NN_CHUNK, MAX_CHAINS = 4, 4, 80
        return
    if name == "colab_min":                                   # the mac_min scope with GPU-sized chunks
        apply_profile("mac_min")
        PROFILE, DESIGN = name, "v1.4-colab_min"
        XGB_WORKERS = None
        DIAG_CHUNK, NN_CHUNK, MAX_CHAINS = 32, 24, 6000
        return
    if name == "t4_min":                                      # colab_min cut further for a T4
        apply_profile("colab_min")
        PROFILE, DESIGN = name, "v1.4-t4_min"
        PASS1_REPS = dict(A=5, A_ref=20, B=5, B_ref=10, R=5)
        A_SLICE_REPS, A_LONG_REPS = 10, 3
        A_POOLING_CELLS = dict(regime=("gradual",), alpha=("2",), size=((10, 60), (30, 120)))
        A_HORSESHOE_CELLS = dict(alpha=("2",), size=((10, 60), (30, 120)))
        B_M2_STRIDE = 1                                        # M2 at every origin (§5.3 as written); 2 = every 12 months
        return
    raise ValueError(f"unknown profile {name}")
