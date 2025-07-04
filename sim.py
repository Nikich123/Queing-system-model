import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import random
import matplotlib.pyplot as plt

#RANDOM_SEED = 123
#.seed(RANDOM_SEED)
#np.random.seed(RANDOM_SEED)

class FuzzyStrategy:
    def __init__(self):

        q = ctrl.Antecedent(np.arange(0, 16, 1), 'queue_length')
        w = ctrl.Antecedent(np.arange(0, 21, 1), 'wait_time')
        d = ctrl.Consequent(np.arange(-3, 4, 1), 'delta_servers')


        q['short'] = fuzz.trapmf(q.universe, [0, 0, 3, 5])
        q['medium'] = fuzz.trapmf(q.universe, [4, 7, 9, 11])
        q['long'] = fuzz.trapmf(q.universe, [10, 12, 15, 15])

        w['low'] = fuzz.trapmf(w.universe, [0, 0, 6, 10])
        w['medium'] = fuzz.trapmf(w.universe, [8, 12, 14, 17])
        w['high'] = fuzz.trapmf(w.universe, [15, 17, 20, 20])

        d['decrease'] = fuzz.trimf(d.universe, [-3, -2, -1])
        d['no_change'] = fuzz.trimf(d.universe, [-1, 0, 1])
        d['increase'] = fuzz.trimf(d.universe, [1, 2, 3])


        rules = [
            ctrl.Rule(q['long'] | w['high'], d['increase']),
            ctrl.Rule(q['medium'] & w['high'], d['increase']),
            ctrl.Rule(q['medium'] & w['medium'], d['no_change']),
            ctrl.Rule(q['short'] & w['low'], d['decrease']),
            ctrl.Rule(q['short'] & w['medium'], d['decrease']),
            ctrl.Rule(q['medium'] & w['low'], d['decrease']),
            ctrl.Rule(q['long'] & w['low'], d['no_change']),
            ctrl.Rule(q['long'] & w['medium'], d['increase']),
            ctrl.Rule(q['short'] & w['high'], d['increase'])
        ]
        system = ctrl.ControlSystem(rules)
        self.sim = ctrl.ControlSystemSimulation(system)

    def get_server_change(self, queue_len, avg_wait):
        self.sim.input['queue_length'] = queue_len
        self.sim.input['wait_time'] = avg_wait
        self.sim.compute()
        return round(self.sim.output['delta_servers'])

class SimpleStrategy:
    def __init__(self, threshold=7):
        self.threshold = threshold
        self.counter = 0

    def get_server_change(self, queue_len, avg_wait):
        if queue_len > self.threshold:
            self.counter += 1
            if self.counter >= 3:
                self.counter = 0
                return 1
            return 1
        elif queue_len < self.threshold:
            self.counter += 1
            return -1
        else:
            self.counter = 0
        return 0

class MaxServersStrategy:
    def get_server_change(self, queue_len, avg_wait):
        return 0

class MassServiceSystem:
    def __init__(self, strategy, sim_time=1000, arrival_rate=1/4, service_rate=1/8,
                 min_servers=1, max_servers=5, abandonment_rate=1/20,
                 c1=500, c2=25, c3=5, c4 = 2):
        self.strategy = strategy
        self.SIM_TIME = sim_time
        self.ARRIVAL_RATE = arrival_rate
        self.SERVICE_RATE = service_rate
        self.MIN_SERVERS = min_servers
        self.MAX_SERVERS = max_servers
        self.ABANDON_RATE = abandonment_rate
        self.C1 = c1
        self.C2 = c2
        self.C3 = c3
        self.C4 = c4

        self.reset()

    def reset(self):
        self.servers = [{'remaining': 0.0, 'car_id': None} for _ in range(self.MIN_SERVERS)]
        self.queue = []
        self.car_patience = {}
        self.wait_times = []
        self.service_durations = []
        self.car_timings = {}
        self.queue_lengths = []
        self.server_counts = []
        self.busy_counts = []
        self.switches = 0
        self.car_id_seq = 0
        self.served_count = 0
        self.abandonments = 0
        self.time = 0
        self.queue_lengths_over_time = []
        self.server_counts_over_time = []

    def run(self):
        self.reset()
        for self.time in range(self.SIM_TIME):

            if random.random() < self.ARRIVAL_RATE:
                self.car_id_seq += 1
                cid = self.car_id_seq
                self.queue.append((cid, self.time))
                self.car_patience[cid] = np.random.exponential(1/self.ABANDON_RATE)
                print(f"Time {self.time}: Car {cid} arrives with patience {self.car_patience[cid]:.2f}")


            remaining = []
            for cid, arrival in self.queue:
                if self.time - arrival > self.car_patience[cid]:
                    self.abandonments += 1
                    print(f"Time {self.time}: Car {cid} abandons after waiting {self.time - arrival:.2f}")
                else:
                    remaining.append((cid, arrival))
            self.queue = remaining


            for server in self.servers:
                if server['remaining'] > 0:
                    server['remaining'] -= 1.0
                    if server['remaining'] <= 0:
                        cid = server['car_id']
                        print(f"Time {self.time}: Car {cid} finished service ({self.car_timings[cid]['service']:.2f})")
                        self.served_count += 1
                        self.service_durations.append(self.car_timings[cid]['service'])
                        server['car_id'] = None
                elif self.queue:
                    cid, arrival = self.queue.pop(0)
                    wait = self.time - arrival
                    self.wait_times.append(wait)
                    raw = np.random.exponential(1 / self.SERVICE_RATE)
                    service = 1 + raw / (raw + 1) * 19
                    self.car_timings[cid] = {'wait': wait, 'service': service}
                    server['car_id'] = cid
                    server['remaining'] = service
                    print(f"Time {self.time}: Car {cid} starts service on server → {self.servers.index(server)+1} (wait {wait:.2f}, service {service:.2f})")


            busy = sum(1 for s in self.servers if s['remaining'] > 0)
            self.busy_counts.append(busy)
            if self.time % 2 == 0:
                q_len = len(self.queue)
                avg_wait = np.mean(self.wait_times[-q_len:]) if q_len and self.wait_times else 0
                delta = self.strategy.get_server_change(q_len, avg_wait)
                desired = max(self.MIN_SERVERS, min(self.MAX_SERVERS, len(self.servers) + delta))
                new_n = desired if desired >= busy else len(self.servers)
                if new_n != len(self.servers):
                    print(f"Time {self.time}: Switching servers {len(self.servers)} -> {new_n}")
                    self.switches += 1
                    if new_n > len(self.servers):
                        self.servers.extend([{'remaining': 0.0, 'car_id': None} for _ in range(new_n - len(self.servers))])
                    else:
                        self.servers = self.servers[:new_n]


            self.queue_lengths.append(len(self.queue))
            self.server_counts.append(len(self.servers))
            self.queue_lengths_over_time.append(len(self.queue))
            self.server_counts_over_time.append(len(self.servers))

        summary = self.summarize()
        print("\n=== Simulation Summary ===")

        for k, v in summary.items():
            if k != 'Pn': print(f"{k}: {v}")
        print(f"Pn state probabilities: {summary['Pn']}\n")
        return summary

    def summarize(self):
        rho = self.ARRIVAL_RATE / self.SERVICE_RATE
        utilization = np.mean(self.busy_counts) / np.mean(self.server_counts) if self.server_counts else 0
        total_server_minutes = sum(self.server_counts)
        # empirical
        Lq = np.mean(self.queue_lengths)
        L = Lq + np.mean(self.busy_counts)
        Wq = np.mean(self.wait_times) if self.wait_times else 0
        W = Wq + (np.mean(self.service_durations) if self.service_durations else 1/self.SERVICE_RATE)
        P0 = sum(1 for q, b in zip(self.queue_lengths, self.busy_counts) if q==0 and b==0) / len(self.queue_lengths)
        P_wait = len([w for w in self.wait_times if w>0]) / len(self.wait_times) if self.wait_times else 0
        sizes = [q + b for q, b in zip(self.queue_lengths, self.busy_counts)]
        unique, counts = np.unique(sizes, return_counts=True)
        Pn = dict(zip(unique.tolist(), (counts/len(sizes)).tolist()))
        quality = self.C1 * self.served_count - self.C2 * total_server_minutes - self.C3 * self.switches - self.C4 * self.abandonments
        return {
            'rho': rho,
            'utilization': utilization,
            'Lq': Lq,
            'L': L,
            'Wq': Wq,
            'W': W,
            'P0': P0,
            'P_wait': P_wait,
            'Pn': Pn,
            'served': self.served_count,
            'abandons': self.abandonments,
            'switches': self.switches,
            'quality': quality,
            'server_minutes': total_server_minutes
        }

if __name__ == "__main__":
    configs = [
        ('Fuzzy', FuzzyStrategy(), {}),
        ('Simple', SimpleStrategy(), {}),
        ('Max', MaxServersStrategy(), {'min_servers':5, 'max_servers':5})
    ]
    results = {}
    for name, strat, overrides in configs:
        print(f"\n=== Running {name} Strategy ===")
        params = {'strategy': strat, 'abandonment_rate':1/20}
        params.update(overrides)
        system = MassServiceSystem(**params)
        results[name] = system.run()

    print("\n=== Comparative Summary ===")
    header = ['Metric'] + list(results.keys())
    metrics_keys = ['served', 'abandons', 'switches', 'rho', 'utilization', 'Lq', 'L', 'Wq', 'W', 'P0', 'P_wait', 'quality', 'server_minutes']
    print(" & ".join(['Metric'] + list(results.keys())))
    for key in metrics_keys:
        row = [key] + [f"{results[name][key]:.3f}" for name in results]
        print(" & ".join(row))
