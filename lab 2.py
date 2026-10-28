import time
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
 
# ---------------------------------------------------------------- parameters
POP_SIZE = 300
GENOME_LEN = 80
GENERATIONS = 50
TARGET_ONES = 50
MUTATION_RATE = 0.01      # probability of flipping each bit
CROSSOVER_RATE = 0.9      # probability that two parents are recombined
ELITES = 2                # best individuals copied unchanged to next generation
TOURNAMENT_K = 3          # individuals compared in each tournament
 
 
# ------------------------------------------------------------ genetic algorithm
def fitness(pop):
    """Fitness of every individual (rows of 0/1).
 
    The value is highest (= chromosome length, 80 by default) when the number
    of ones equals 50 and drops by 1 for every one above or below 50.
    """
    pop = np.atleast_2d(pop)
    return pop.shape[1] - np.abs(pop.sum(axis=1) - TARGET_ONES)
 
 
def run_ga(pop_size, length, generations, mutation_rate, seed):
    """Run the GA. Returns (trace DataFrame, ones count of every individual per generation, final population)."""
    rng = np.random.default_rng(seed)
    pop = rng.integers(0, 2, size=(pop_size, length))      # random initial population
    trace, ones_by_gen = [], []
 
    for gen in range(1, generations + 1):
        fits = fitness(pop)
        order = np.argsort(-fits, kind='stable')            # fittest first
        best = pop[order[0]]
        trace.append({
            'generation': gen,
            'best': int(fits.max()),
            'average': round(float(fits.mean()), 2),
            'worst': int(fits.min()),
            'ones in best': int(best.sum()),
            'best pattern': ''.join(map(str, best)),
        })
        ones_by_gen.append(pop.sum(axis=1))
        if gen == generations:
            break
 
        # 1) selection: tournament of K random individuals for each parent
        n_children = pop_size - ELITES
        candidates = rng.integers(0, pop_size, size=(2, n_children, TOURNAMENT_K))
        winners = np.take_along_axis(candidates, fits[candidates].argmax(axis=2)[..., None], axis=2)[..., 0]
        parent1, parent2 = pop[winners[0]], pop[winners[1]]
 
        # 2) single-point crossover (with probability CROSSOVER_RATE)
        cut = rng.integers(1, length, size=n_children)
        recombine = rng.random(n_children) < CROSSOVER_RATE
        take_parent1 = (np.arange(length)[None, :] < cut[:, None]) | ~recombine[:, None]
        children = np.where(take_parent1, parent1, parent2)
 
        # 3) mutation: flip each bit with a small probability
        flip = rng.random(children.shape) < mutation_rate
        children = np.where(flip, 1 - children, children)
 
        # 4) elitism: keep the best individuals, fill the rest with children
        pop = np.vstack([pop[order[:ELITES]], children])
 
    return pd.DataFrame(trace), np.array(ones_by_gen), pop
 
 
def draw_generation(trace, ones_by_gen, step, length):
    """Left: best / average / worst fitness up to this generation. Right: ones per individual in this generation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    shown = trace.iloc[:step]
    ax1.plot(shown['generation'], shown['best'], color='#10b981', label='Best')
    ax1.plot(shown['generation'], shown['average'], color='#3b82f6', label='Average')
    ax1.plot(shown['generation'], shown['worst'], color='#ef4444', label='Worst')
    ax1.set_xlim(1, len(trace))
    ax1.set_ylim(max(0, trace['worst'].min() - 3), length + 2)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Fitness over generations')
    ax1.legend(loc='lower right')
 
    low = min(ones_by_gen.min(), TARGET_ONES - 3) - 1
    high = max(ones_by_gen.max(), TARGET_ONES + 3) + 1
    ax2.hist(ones_by_gen[step - 1], bins=np.arange(low, high + 2) - 0.5, color='#93c5fd', edgecolor='white')
    ax2.axvline(TARGET_ONES, color='#10b981', linestyle='--', label='Target (50 ones)')
    ax2.set_xlim(low, high)
    ax2.set_xlabel('Number of ones in an individual')
    ax2.set_ylabel('Individuals')
    ax2.set_title(f'Population at generation {step}')
    ax2.legend(loc='upper left')
    fig.tight_layout()
    return fig
 
 
# ======== STREAMLIT APP ========
cached_run_ga = st.cache_data(show_spinner=False)(run_ga)
 
# reset button
DEFAULTS = {'pop_size': POP_SIZE, 'length': GENOME_LEN, 'generations': GENERATIONS,
            'mutation': MUTATION_RATE, 'seed': 42, 'speed': 0.2}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
 
 
def reset_defaults():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    for key in [k for k in st.session_state if k.startswith('step_')]:
        del st.session_state[key]  # generation slider goes back to the last generation
 
 
st.set_page_config(page_title='BSD3513 | Lab Report 2', page_icon='🧬', layout='wide')
st.title('LAB REPORT 2: Genetic Algorithm | Interactive Bit Pattern Evolution')
st.caption('BSD3513 Artificial Intelligence · SD24048 WONG JIA YING · Genetic Algorithm · Target: 50 ones in an 80-bit pattern · UMPSA')
 
with st.expander('Learning guide', expanded=False):
    st.markdown(
        'A **genetic algorithm** is a search heuristic inspired by natural selection. A population of random bit '
        'patterns (chromosomes) is scored by a **fitness function**; fitter individuals are more likely to be '
        'selected as parents (**tournament selection**), their bits are recombined (**single-point crossover**) '
        'and randomly flipped (**mutation**), and the offspring replace the less fit chromosomes. The best '
        'individuals are always kept (**elitism**).\n\n'
        'Here the fitness is **length − |ones − 50|**, so an 80-bit pattern scores its maximum of **80** exactly '
        'when it contains **50 ones**.')
 
with st.sidebar:
    st.header('Genetic algorithm settings')
    st.slider('Population size', 50, 1000, step=50, key='pop_size')
    st.number_input('Chromosome length', 60, 200, step=1, key='length')
    st.slider('Number of generations', 10, 200, key='generations')
    st.slider('Mutation rate (per bit)', 0.0, 0.1, step=0.005, key='mutation')
    st.number_input('Random seed', 0, 9999, step=1, key='seed')
    st.slider('Autoplay speed (seconds per generation)', 0.05, 1.0, step=0.05, key='speed')
    st.caption('Defaults are the values required by the case study (300 / 80 / 50).')
    st.button('Reset to default settings', on_click=reset_defaults, width='stretch')
 
pop_size = int(st.session_state.pop_size)
length = int(st.session_state.length)
generations = int(st.session_state.generations)
mutation = float(st.session_state.mutation)
seed = int(st.session_state.seed)
speed = st.session_state.speed
 
trace, ones_by_gen, final_pop = cached_run_ga(pop_size, length, generations, mutation, seed)
last = trace.iloc[-1]
reached = trace[trace['best'] == length]
 
# ---------------------------------------------------------------- results
st.subheader('Result')
a, b, c, d = st.columns(4)
a.metric('Best fitness', int(last['best']))
b.metric('Ones in best pattern', int(last['ones in best']))
c.metric('First generation at maximum', int(reached['generation'].iloc[0]) if len(reached) else '—')
d.metric('Final average fitness', f"{last['average']:.2f}")
st.write('**Best bit pattern found**')
st.code(last['best pattern'])
if last['best'] == length:
    st.success(f'The best pattern contains exactly {TARGET_ONES} ones, so it returns the maximum fitness of {length}.')
else:
    st.warning('The maximum fitness was not reached with these settings. Try more generations or a different seed.')
 
st.divider()
 
# ---------------------------------------------------------------- evolution
st.subheader('Evolution step by step')
step = st.slider('Inspect generation', 1, generations, generations,
                 key=f'step_{generations}_{pop_size}_{length}_{mutation}_{seed}')
st.caption('💡 Click **Autoplay** to watch the population evolve generation by generation.')
play = st.button('▶ Autoplay evolution')
 
left, right = st.columns([1.6, 1])
with left:
    graph_box = st.empty()
    info_box = st.empty()
with right:
    st.write('**Generation trace**')
    st.dataframe(trace.drop(columns='best pattern'), hide_index=True, width='stretch')
 
for s in (range(1, generations + 1) if play else [step]):
    fig = draw_generation(trace, ones_by_gen, s, length)
    graph_box.pyplot(fig)
    plt.close(fig)
    row = trace.iloc[s - 1]
    info_box.info(f"Generation {s} of {generations} · best fitness: {row['best']} · "
                  f"average: {row['average']:.2f} · ones in best: {row['ones in best']}\n\n`{row['best pattern']}`")
    if play:
        time.sleep(speed)
 
with st.expander('Best pattern in every generation'):
    st.dataframe(trace[['generation', 'best', 'ones in best', 'best pattern']], hide_index=True, width='stretch')
 
st.divider()
 
# ---------------------------------------------------------------- fitness function
st.subheader('Fitness function')
ones = np.arange(0, length + 1)
fit_values = length - np.abs(ones - TARGET_ONES)
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(ones, fit_values, color='#3b82f6')
ax.scatter([TARGET_ONES], [length], color='#10b981', zorder=3, label=f'Maximum = {length} at {TARGET_ONES} ones')
ax.set_xlabel('Number of ones')
ax.set_ylabel('Fitness')
ax.legend()
st.pyplot(fig)
plt.close(fig)
with st.expander('Click here to display fitness values'):
    st.dataframe(pd.DataFrame({'ones': ones, 'fitness': fit_values}), hide_index=True)
 
st.divider()
 
# ---------------------------------------------------------------- case-study check
st.subheader('Case-study requirement check')
checks = [
    ('Population size', POP_SIZE, pop_size),
    ('Length of each individual', GENOME_LEN, length),
    ('Number of generations', GENERATIONS, generations),
    ('Number of ones giving the maximum', TARGET_ONES, int(last['ones in best']) if last['best'] == length else '—'),
    ('Return value when ones = 50', 80, int(fitness(np.array([[1] * TARGET_ONES + [0] * (length - TARGET_ONES)]))[0])),
]
check_df = pd.DataFrame([(n, str(req), str(cur), '✔' if req == cur else '✘') for n, req, cur in checks],
                        columns=['Requirement', 'Required', 'Current run', 'Met'])
st.dataframe(check_df, hide_index=True, width='stretch')
st.caption('A ✘ only means the sidebar has been changed away from the case-study values; press Reset to return to them.')
 
st.divider()
 
# ---------------------------------------------------------------- final population
st.subheader('Final population')
bit_freq = final_pop.mean(axis=0)
fig, ax = plt.subplots(figsize=(10, 3))
ax.bar(np.arange(1, length + 1), bit_freq, color='#93c5fd')
ax.axhline(TARGET_ONES / length, color='#10b981', linestyle='--', label=f'{TARGET_ONES}/{length} = {TARGET_ONES / length:.3f}')
ax.set_xlabel('Bit position')
ax.set_ylabel('Proportion of ones')
ax.set_ylim(0, 1)
ax.legend()
st.pyplot(fig)
plt.close(fig)
st.caption('Fitness only depends on how many ones a pattern has, not where they are, so every bit position drifts '
           'towards a proportion of about 50/80 = 0.625.')