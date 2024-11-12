
from simulator import Simulator
sim = Simulator(num_nodes = 10, rho = 1.1, cobra_walk_rho=0.8)
sim.setup()
sim.show_network()
results = sim.run()
print(results)