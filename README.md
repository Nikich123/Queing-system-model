This project simulates a mass service system (e.g., a car service center or call center) with dynamically adjustable server count. It compares three different server allocation strategies based on system load:

Fuzzy Logic Strategy: Uses fuzzy inference to evaluate queue length and average wait time, then adaptively adjusts the number of servers.

Simple Strategy: Adds or removes servers based on a queue length threshold and persistence.

Static Strategy: Keeps the number of servers constant (maximum capacity).
The code contains discrete-event simulation of customer arrivals, patience (abandonment), service, and queuing, dynamic adjustment of server count in response to system state.
Tracks key performance metrics: queue length, wait times, server utilization, abandonment rate, and quality score.

The work explores comparative analysis and visualizations across multiple simulation runs.

sim.py: Contains the main simulation logic, strategies, and metric computations.

sim_multiple.py: Runs multiple simulations per strategy, aggregates metrics, and generates comparison plots in a PDF report.

Output :

Queue statistics (Lq, Wq), system stats (L, W), server utilization, abandonment count, service quality, etc.

Probability distribution of system states (P(n)).

Automatically generated report results.pdf with plots showing:

Probability of empty system and queue presence

Average queue length vs. wait time

Average system length vs. system time

Libraries required: numopy, random, skfuzzy.

How to run a simulation?
Run a comparative simulation with 100 runs per strategy and generate plots:
sim_multiple.py
Or run a single-shot simulation:
sim.py

The results are in Ukrainian language.
