import random
import numpy as np
import statistics
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sim import (
    FuzzyStrategy,
    SimpleStrategy,
    MaxServersStrategy,
    MassServiceSystem
)
#pdf
def main(num_runs=10, output_pdf='results.pdf'):
    configs = {
        'Fuzzy':  (FuzzyStrategy,      {'abandonment_rate':1/20}),
        'Simple': (SimpleStrategy,     {'abandonment_rate':1/20}),
        'Max':    (MaxServersStrategy, {'min_servers': 5, 'max_servers': 5, 'abandonment_rate':1/20})
    }


    sample_summary = MassServiceSystem(strategy=FuzzyStrategy(), abandonment_rate=1/20).run()
    metrics_keys = list(sample_summary.keys())


    aggregate = { name: {k: [] for k in metrics_keys if k != 'Pn'} for name in configs }
    aggregate_Pn = { name: [] for name in configs }


    for name, (StratClass, overrides) in configs.items():
        for _ in range(num_runs):
            seed = random.randint(1, 100_000_000)
            random.seed(seed)
            np.random.seed(seed)

            strat  = StratClass()
            system = MassServiceSystem(strategy=strat, **overrides)
            summary = system.run()

            for k, v in summary.items():
                if k == 'Pn':
                    aggregate_Pn[name].append(v)
                else:
                    aggregate[name][k].append(v)


    avg_metrics = { name: {k: statistics.mean(vals) for k, vals in aggregate[name].items()} for name in configs }


    avg_pn = {}
    for name in configs:
        pn_dicts = aggregate_Pn[name]
        all_states = sorted(set().union(*pn_dicts))
        avg_pn[name] = { state: statistics.mean([d.get(state, 0.0) for d in pn_dicts]) for state in all_states }


    with PdfPages(output_pdf) as pdf:
        for name in configs:
            servers = configs[name][1].get('max_servers', 1)

            P0 = avg_pn[name].get(0, 0)
            Pq = sum(p for s, p in avg_pn[name].items() if s > servers)
            fig1, ax1 = plt.subplots()
            ax1.bar(['Empty System', 'Customers In Queue'], [P0, Pq], color=['skyblue', 'salmon'])
            ax1.set_ylim(0, 1)
            ax1.set_ylabel('Вірогідність')
            ax1.set_title(f'Стратегія {name}: Ймовірність пустої системи і клієнтів у черзі ')
            pdf.savefig(fig1)
            plt.close(fig1)


            Wq = avg_metrics[name]['Wq']
            Lq = avg_metrics[name]['Lq']
            fig2, ax2 = plt.subplots()
            ax2.bar(['Wq', 'Lq'], [Wq, Lq], color=['lightgreen', 'orange'])
            ax2.set_ylabel('Середнє значення')
            ax2.set_title(f'Стратегія {name}: Час проведений у черзі і довжина черги')
            pdf.savefig(fig2)
            plt.close(fig2)


            Ws = avg_metrics[name]['W']
            Ls = avg_metrics[name]['L']
            fig3, ax3 = plt.subplots()
            ax3.bar(['W', 'L'], [Ws, Ls], color=['plum', 'gold'])
            ax3.set_ylabel('Середнє значення')
            ax3.set_title(f'Стратегія {name} : Час проведений у системі і довжина системи')
            pdf.savefig(fig3)
            plt.close(fig3)

    print(f"Plots saved to {output_pdf}\n")


    for name in configs:
        print(f"--- {name} Strategy Summary ---")
        print("Average Metrics:")
        for k, v in avg_metrics[name].items():
            print(f"  {k}: {v:.4f}")
        print("Average State Probabilities (Pn):")
        for state, prob in sorted(avg_pn[name].items()):
            print(f"  P({state}) = {prob:.4f}")
        print()

if __name__ == '__main__':
    main(num_runs=100)
