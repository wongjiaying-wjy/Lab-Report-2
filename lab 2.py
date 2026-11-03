import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
 
# ---- Case-study defaults ----
POP_SIZE = 300            # 1) population setup
CHROM_LEN = 80            # 3) length of all individuals
TARGET_ONES = 50          # 2) fitness peaks when number of ones == 50
MAX_FITNESS = 80          # 4) return value is 80 when ones == 50
N_GENERATIONS = 50        # 5) number of generations
 
SELECTION_METHODS = ["Tournament", "Roulette wheel", "Rank-based"]
CROSSOVER_METHODS = ["One-point", "Two-point", "Uniform"]
 
# Default values of the adjustable settings
DEFAULTS = {"sel": "Tournament", "cx": "One-point", "cxr": 0.9, "mut": 0.0125, "seed": 42}
 
# ---- Fixed GA settings ----
TOURNAMENT_K = 3
ELITISM = 2               # best individuals copied unchanged to next generation
 
 
# ---- Fitness Function ----
def fitness(individual: np.ndarray, max_fitness: int) -> float:
    """Maximum (80) when ones == 50.
    Every one above or below 50 costs 1 point."""
    ones = int(individual.sum())
    return max_fitness - abs(ones - TARGET_ONES)
 
 
# ---- GA Operators ----
def init_population(pop_size, chrom_len):
    return np.random.randint(0, 2, size=(pop_size, chrom_len), dtype=np.int8)
 
 
def tournament_selection(pop, fits, k):
    idxs = np.random.randint(0, len(pop), size=k)
    return pop[idxs[np.argmax(fits[idxs])]].copy()
 
 
def selection_probabilities(fits, method):
    """Roulette wheel: chance proportional to fitness.
    Rank-based: chance proportional to rank (worst = 1, best = population size)."""
    if method == "Roulette wheel":
        return fits / fits.sum()
    if method == "Rank-based":
        ranks = np.empty(len(fits))
        ranks[np.argsort(fits, kind="stable")] = np.arange(1, len(fits) + 1)
        return ranks / ranks.sum()
    return None
 
 
def select_parent(pop, fits, method, probs):
    if method == "Tournament":
        return tournament_selection(pop, fits, TOURNAMENT_K)
    return pop[np.random.choice(len(pop), p=probs)].copy()
 
 
def one_point_crossover(p1, p2):
    point = np.random.randint(1, len(p1))
    return (np.concatenate([p1[:point], p2[point:]]),
            np.concatenate([p2[:point], p1[point:]]))
 
 
def two_point_crossover(p1, p2):
    a, b = sorted(np.random.choice(range(1, len(p1)), size=2, replace=False))
    return (np.concatenate([p1[:a], p2[a:b], p1[b:]]),
            np.concatenate([p2[:a], p1[a:b], p2[b:]]))
 
 
def uniform_crossover(p1, p2):
    mask = np.random.rand(len(p1)) < 0.5
    return np.where(mask, p1, p2), np.where(mask, p2, p1)
 
 
def crossover(p1, p2, method, rate):
    if np.random.rand() > rate:
        return p1.copy(), p2.copy()
    if method == "Two-point":
        return two_point_crossover(p1, p2)
    if method == "Uniform":
        return uniform_crossover(p1, p2)
    return one_point_crossover(p1, p2)
 
 
def mutate(individual, rate):
    mask = np.random.rand(len(individual)) < rate
    individual[mask] = 1 - individual[mask]
    return individual
 
 
def evolve(pop, generations, mut_rate, max_fitness, selection, cx_method, cx_rate):
    history = []
    best_individual, best_f, first_hit = None, -np.inf, None
 
    for gen in range(1, generations + 1):
        fits = np.array([fitness(ind, max_fitness) for ind in pop], dtype=float)
        ones = pop.sum(axis=1)
 
        # Record statistics for this generation
        idx = int(np.argmax(fits))
        history.append({
            "Generation": gen,
            "Best fitness": float(fits[idx]),
            "Average fitness": float(fits.mean()),
            "Worst fitness": float(fits.min()),
            "Average ones": float(ones.mean()),
            "Individuals with 50 ones": int((ones == TARGET_ONES).sum()),
        })
        if fits[idx] > best_f:
            best_f, best_individual = float(fits[idx]), pop[idx].copy()
        if first_hit is None and fits[idx] == max_fitness:
            first_hit = gen
 
        # Elitism: keep the best individuals unchanged
        elite_idx = np.argsort(fits, kind="stable")[-ELITISM:]
        new_pop = [pop[i].copy() for i in elite_idx]
 
        # Fill the rest: selection -> crossover -> mutation
        probs = selection_probabilities(fits, selection)
        while len(new_pop) < len(pop):
            p1 = select_parent(pop, fits, selection, probs)
            p2 = select_parent(pop, fits, selection, probs)
            c1, c2 = crossover(p1, p2, cx_method, cx_rate)
            new_pop.extend([mutate(c1, mut_rate), mutate(c2, mut_rate)])
        pop = np.array(new_pop[:len(pop)], dtype=np.int8)
 
    return best_individual, best_f, pd.DataFrame(history), pop, first_hit
 
 
def reset_defaults():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.session_state.pop("result", None)
 
 
# ---- Streamlit UI ----
st.set_page_config(page_title="Genetic Algorithm: Bit Pattern Evolution", page_icon="🧬", layout="centered")
 
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
 
st.title("🧬 Genetic Algorithm: Bit Pattern Evolution")
st.caption("BSD3513 Introduction to AI · Lab Report 2 · SD24048 WONG JIA YING")
 
# Fixed requirements (cannot be changed)
st.subheader("ℹ️ Problem setup")
st.caption("Fixed by the lab requirement")
st.table(pd.DataFrame({
    "Requirement": ["Population size", "Chromosome length", "Target number of ones",
                    "Max fitness at optimum", "Generations"],
    "Value": [POP_SIZE, CHROM_LEN, TARGET_ONES, MAX_FITNESS, N_GENERATIONS],
}).set_index("Requirement"))
st.markdown("### Fitness function")
st.latex(rf"f(x) = {MAX_FITNESS} - \left|\, \text{{ones}}(x) - {TARGET_ONES} \,\right|")
st.markdown(
    f"The maximum fitness is **{MAX_FITNESS}**, reached when the pattern has exactly "
    f"**{TARGET_ONES}** ones. Every one above or below {TARGET_ONES} costs 1 point."
)
 
# Adjustable settings, above the Run button
st.subheader("Genetic Algorithm Settings")
col1, col2, col3 = st.columns(3)
with col1:
    selection = st.selectbox(
        "Selection method", SELECTION_METHODS, key="sel",
        help="How parents are chosen. Tournament: the best of 3 random individuals wins. "
             "Roulette wheel: chance proportional to fitness. "
             "Rank-based: chance proportional to rank (best = highest).")
with col2:
    cx_method = st.selectbox(
        "Crossover method", CROSSOVER_METHODS, key="cx",
        help="How two parents mix their bits. One-point: swap the tails after one cut. "
             "Two-point: swap the middle part. Uniform: each bit comes from either parent.")
with col3:
    seed = st.number_input(
        "Random seed", min_value=0, step=1, key="seed",
        help="Same seed gives the same result every time. Change it to try a different run.")
col4, col5 = st.columns(2)
with col4:
    cx_rate = st.slider(
        "Crossover rate", 0.0, 1.0, key="cxr", step=0.05,
        help="Chance that two parents are crossed over. Otherwise the children are copies of the parents.")
with col5:
    mut_rate = st.slider(
        "Mutation rate per bit", 0.0, 0.10, key="mut", step=0.0025, format="%.4f",
        help="Chance that each bit flips (0↔1) in a new child. Higher means more "
             "variety but less stability. Default 1/80 = 0.0125 (about one flip per pattern).")
 
# How to use (above the Run button)
st.subheader("📖 How to use")
st.info(
    "1. The case-study values (300 / 80 / 50) are fixed. You can change the "
    "**selection method**, **crossover method**, **seed**, **crossover rate** and **mutation rate**.\n"
    "2. Click **Run Genetic Algorithm**.\n"
    "3. Read the best pattern and the convergence plot.\n"
    "4. Change the **seed** to get a different random run.\n"
    "5. Click **Reset to defaults** to restore the default settings.\n\n"
    "💡 Hover over the **?** beside a setting for an explanation."
)
 
with st.expander("🧠 How it works"):
    st.markdown(f"""
1. **Initialise** random bit strings (population).
2. **Fitness:** `{MAX_FITNESS} - |ones - {TARGET_ONES}|`.
3. **Selection:** parents are chosen by the selected method (Tournament, Roulette wheel or Rank-based).
4. **Crossover:** two parents mix their bits using the selected method (One-point, Two-point or Uniform).
5. **Mutation:** each bit flips with the chosen mutation rate.
6. **Elitism:** the {ELITISM} best individuals are copied unchanged.
7. Repeat for {N_GENERATIONS} generations and report the best pattern.
 
Cost of one run is **O(G × P × L)**: for the case study 50 × 300 × 80 = 1,200,000 bit operations.
""")
 
_, mid, _ = st.columns([1, 2, 1])
with mid:
    run_clicked = st.button("Run Genetic Algorithm", type="primary", use_container_width=True)
    st.button("Reset to defaults", on_click=reset_defaults, use_container_width=True,
              help="Restore the default settings and clear the previous result.")
 
if run_clicked:
    random.seed(int(seed))
    np.random.seed(int(seed))
    population = init_population(POP_SIZE, CHROM_LEN)
    st.session_state["result"] = (
        MAX_FITNESS, POP_SIZE, CHROM_LEN, N_GENERATIONS, int(seed),
        *evolve(population, N_GENERATIONS, mut_rate, MAX_FITNESS, selection, cx_method, cx_rate),
    )
    st.session_state["used"] = (selection, cx_method, cx_rate, mut_rate)
 
if "result" in st.session_state:
    (max_f, pop_used, len_used, gens_used, seed_used,
     best_ind, best_fit, history, _final_pop, first_hit) = st.session_state["result"]
 
    ones_count = int(best_ind.sum())
    st.subheader("Best Individual Found")
    c1, c2, c3 = st.columns(3)
    c1.metric("Best fitness", f"{best_fit:.0f} / {max_f}")
    c2.metric("Number of ones", ones_count)
    c3.metric("Optimum first reached", f"Gen {first_hit}" if first_hit else "—")
    st.write(f"**Ones**: {ones_count} | **Zeros**: {len_used - ones_count} | "
             f"**Length**: {len_used} | **Seed**: {seed_used}")
    sel_u, cx_u, cxr_u, mut_u = st.session_state["used"]
    st.caption(f"Settings used: {sel_u} selection, {cx_u} crossover (rate {cxr_u:.2f}), "
               f"mutation rate {mut_u:.4f}")
    st.code("".join(map(str, best_ind.tolist())), language="text")
 
    # Convergence plot
    st.subheader("📉 Convergence (Fitness per Generation)")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["Generation"], history["Best fitness"], linewidth=2, label="Best")
    ax.plot(history["Generation"], history["Average fitness"], linewidth=2, linestyle="--", label="Average")
    ax.plot(history["Generation"], history["Worst fitness"], linewidth=1, alpha=0.7, label="Worst")
    ax.axhline(max_f, color="grey", linestyle=":", label=f"Max = {max_f}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("GA Convergence")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("With a large random population, one individual with exactly 50 ones often exists "
               "already in generation 1, so the best line can start at the maximum. "
               "The average line shows the whole population improving.")
 
    with st.expander("📋 Per-generation statistics"):
        st.dataframe(history, hide_index=True)
