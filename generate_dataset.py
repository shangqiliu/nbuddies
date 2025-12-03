from src.ICs import *
from src.evolution import simulation, load_data_pkl
import os
import argparse
import pickle as pkl

# Create the parser
parser = argparse.ArgumentParser(prog="NBuddies mass dataset generation", description="Run many N-body simulations randomly distributed within a parameter space.", 
                                 epilog="Created in Fall 2025 quarter of PHYS 206: Computational Astrophysics at UCR.")

# Add arguments
# Required
parser.add_argument("N", type=int, help="Number of simulations to run")

parser.add_argument("Name", type=str, help="Name of your dataset")

parser.add_argument("--time", type=float, help="Time each should be allowed to run, defaults to 3*relaxation time for the middle of your box", default=None)

parser.add_argument("--M_min", type = int, help = "Minimum BH Mass", default = 1e5)
parser.add_argument("--M_max", type = int, help = "Maximum BH Mass", default = 1e10)

parser.add_argument("--R_min", type = int, help = "Minimum Scale Radius", default = 1e-1)
parser.add_argument("--R_max", type = int, help = "Maximum Scale Radius", default = 1e1)

parser.add_argument("--N_min", type = int, help = "Minimum Number of BHs", default = 50)
parser.add_argument("--N_max", type = int, help = "Maximum Number of BHs", default = 100)

parser.add_argument("--clear", action=argparse.BooleanOptionalAction, 
                    help="Clears all previous data from this dataset", default=False)

# Parse the arguments
args = parser.parse_args()

nbuddies_path = os.path.dirname(os.path.realpath(__file__))
dataset_path = nbuddies_path+"/training_data"

#create directory if non-existent
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

if os.path.exists(dataset_path + "/" + args.Name + ".pkl") and not args.clear:
    with open(dataset_path + "/" + args.Name + ".pkl", 'rb') as f:
        dataset = pkl.load(f)
else:
    dataset = {"seeds" : np.asarray([]), "Ns" : np.asarray([]), "Ms" : np.asarray([]), "Rs" : np.asarray([]), "ICs" : np.asarray([]), "Final_Data" : np.asarray([])}

#prep params
[Ns, Ms, Rs] = np.random.rand(3,args.N)

Ns *= (args.N_max - args.N_min)
Ns += args.N_min
Ns = Ns.astype(int)

Ms *= (np.log(args.M_max) - np.log(args.M_min))
Ms += np.log(args.M_min)
Ms = np.exp(Ms)

Rs *= (np.log(args.R_max) - np.log(args.R_min))
Rs += np.log(args.R_min)
Rs = np.exp(Rs)

if args.time is None:
    N_ave = 0.5 * (args.N_max + args.N_min)
    R_ave = 0.5 * (args.R_max + args.R_min)
    M_ave = 0.5 * (args.M_max + args.M_min)
    kpc_per_km = 3.0856775814671916e16
    t_relax = 0.14 * N_ave * (R_ave**(3/2)) / (np.log(0.4*N_ave) * np.sqrt(GG*M_ave)) * kpc_per_km
    sim_time = 3*t_relax
else:
    sim_time = args.time

print(sim_time)

def _find_last_batch_num(sim_name) -> int:
    """
    finds num of last batch file saved

    Parameters
    ----------
    sim_name : str
        The name of the simulation
    Returns
    -------
    int
        num of last batch file saved
    """

    i = 0
    while os.path.exists(nbuddies_path + "/data/" + sim_name + f"/data_batch{i}.pkl"): # while path of ith data batch exists
        i += 1 # increment i
    return i - 1 # i is number corresponding to last data batch number

for n in range(args.N):
    data_path = nbuddies_path+"/data/"+args.Name+f"_{len(dataset['Ns'])}"
    #create directory if non-existent
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    BHs, _ = generate_plummer_initial_conditions(
        n_blackholes=Ns[n],
        initial_mass=Ms[n],
        scale=Rs[n],
        ratio=0
    )
    print(
        f"Running {args.Name}_{n} with: N={Ns[n]}, R={Rs[n]}, M={Ms[n]}"
    )

    pkl.dump(BHs, open(data_path+"/ICs.pkl", "wb"))
    
    simulation(data_path+"/ICs.pkl", data_path, tot_time = sim_time, nsteps = 20, delta_t = None,
           adaptive_dt = True, eta = 0.1, leapfrog = True, use_tree = True,
           use_dynamic_criterion = True, ALPHA = 0.1, THETA_0 = None)
    
    last_batch_num = _find_last_batch_num(args.Name+f"_{len(dataset['Ns'])}")
    with open(nbuddies_path + '/data/' + args.Name + f"_{len(dataset['Ns'])}/data_batch{last_batch_num}.pkl", 'rb') as file:
        file = pkl.load(file)
        data = file["data"][-1]

    #appends after each run so output is accurate even if stopped early
    rand_state = np.random.get_state()
    dataset["seeds"] = np.append(dataset["seeds"], {"algorithm" : rand_state[0], "keys" : rand_state[1], "pos" : rand_state[2], "has_gauss" : rand_state[3], "cached_gaussian" : rand_state[4]})
      
    if len(dataset["ICs"]) == 0:
        dataset["ICs"] = np.asarray([BHs])
    else:
        dataset["ICs"] = np.append(dataset["ICs"], BHs)
    
    if len(dataset["Final_Data"]) == 0:
        dataset["Final_Data"] = np.asarray([{"data": data}])
    else:
        dataset["Final_Data"] = np.append(dataset["Final_Data"], {"data": data})
    
    dataset["Ns"] = np.append(dataset["Ns"], Ns[n])
    dataset["Ms"] = np.append(dataset["Ms"], Ms[n])
    dataset["Rs"] = np.append(dataset["Rs"], Rs[n])

    with open(dataset_path + "/" + args.Name + ".pkl", 'wb') as f:
        pkl.dump(dataset, f)