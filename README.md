# 🧬 Genetic Algorithm Explorer: Evolving an 80-bit Pattern with 50 Ones

**BSD3513 Introduction to Artificial Intelligence · Lab Report 2**

**SD24048 WONG JIA YING**

🌐 **Live App:** [https://lab-report-2-pqeussphud9jfdbura28ye.streamlit.app/](https://lab-report-2-pqeussphud9jfdbura28ye.streamlit.app/)

💻 **Source Code:** [[(https://github.com/wongjiaying-wjy/Lab-Report-2)]]

![App overview](image/app_overview.png)

## About

An interactive Streamlit application that uses a genetic algorithm to evolve a bit pattern with predefined parameters from the case study: a population of **300** individuals, each **80** bits long, evolved for **50** generations. The fitness function returns its maximum value of **80** when the number of ones equals **50**. The app shows the best pattern found, how the fitness changes over the generations, and how the population gathers around 50 ones.

## Key Features

- Case-study settings (300 / 80 / 50) by default, with sliders for population size, chromosome length, generations, mutation rate and random seed.
- Result summary with the best bit pattern, its number of ones and fitness, and the first generation that reached the maximum.
- Explore the evolution using the generation slider.
- Watch the population evolve using Autoplay.
- Generation trace table, fitness-function plot and final-population bit-position chart.
- Case-study requirement check table.
- Reset to the default settings.

## Screenshots

| Generation 1 | Generation 50 |
|---|---|
| ![Generation 1](image/generation_1.png) | ![Generation 50](image/generation_50.png) |

**Fitness function and requirement check**

![Fitness function](image/fitness_function.png)

## How to Run

```bash
pip install -r requirements.txt
streamlit run SD24048_Lab2.py
```

The app usually opens at <http://localhost:8501>.

## How the Genetic Algorithm Works

1. Create 300 random 80-bit chromosomes.
2. Score every chromosome with `fitness = length - |ones - 50|`.
3. **Selection:** a tournament of 3 random individuals picks each parent.
4. **Crossover:** single-point crossover with probability 0.9.
5. **Mutation:** each bit flips with probability 0.01.
6. **Elitism:** the 2 best individuals are copied unchanged into the next generation.
7. Repeat for 50 generations and report the best pattern.

## Code Overview

| Function | Purpose |
|---|---|
| `fitness()` | Scores each chromosome; maximum (80) when it has 50 ones. |
| `run_ga()` | Runs selection, crossover, mutation and elitism for all generations. |
| `draw_generation()` | Draws the fitness curves and the number-of-ones histogram for a generation. |
| `reset_defaults()` | Restores default settings. |

## Repository Structure

```
Lab-Report-2/
├── SD24048_Lab2.py
├── requirements.txt
├── README.md
└── image/
```

## Built with

Python · Streamlit · NumPy · Matplotlib · pandas

## Complexity note

Each generation evaluates, selects, recombines and mutates *P* individuals of length *L*, so one run costs **O(G × P × L)** operations. For the case study this is 50 × 300 × 80 = 1,200,000 bit operations. The genetic algorithm is a heuristic: it does not guarantee the optimum, but with this simple fitness function it reaches 80 quickly.
