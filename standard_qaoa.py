from mindquantum.core import Circuit, Hamiltonian, UN, H, ZZ, RX, QubitOperator
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum.simulator import Simulator
import networkx as nx
import numpy
import mindspore.nn as nn
import mindspore as ms
import numpy as np
import time
import random
import scipy.optimize as sopt
from mindspore import Tensor

g = nx.Graph()
nx.add_path(g, [0, 1])
nx.add_path(g, [1, 2])
nx.add_path(g, [2, 3])
nx.add_path(g, [3, 4])
nx.add_path(g, [0, 4])
nx.add_path(g, [0, 2])
# nx.draw(g, with_labels=True, font_weight='bold')

for i in g.nodes:
    print('one size:', [i], 'cut=', nx.cut_size(g, [i]))           
    for j in range(i):
        print('one size:', [i, j], 'cut=', nx.cut_size(g, [i, j]))


def build_hc(g, para):
    hc = Circuit()                  
    for i in g.edges:
        hc += ZZ(para).on(i)        
    hc.barrier()                   
    return hc

circuit = build_hc(g, 'gamma')

def build_hb(g, para):
    hb = Circuit()                  
    for i in g.nodes:
        hb += RX(para).on(i)       
    hb.barrier()                  
    return hb

circuit = build_hb(g, 'beta')

circuit = build_hc(g, 'gamma') + build_hb(g, 'beta')

def build_ansatz(g, p):                    
    circ = Circuit()                      
    for i in range(p):
        circ += build_hc(g, f'g{i}')      
        circ += build_hb(g, f'b{i}')       
    return circ

def build_ham(g):
    ham = QubitOperator()
    for i in g.edges:
        ham += QubitOperator(f'Z{i[0]} Z{i[1]}')  
    return ham

p = 4
ham = Hamiltonian(build_ham(g))              
init_state_circ = UN(H, g.nodes)             
ansatz = build_ansatz(g, p)                
circ = init_state_circ + ansatz  


ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


sim = Simulator('projectq', circ.n_qubits)                    

grad_ops = sim.get_expectation_with_grad(ham, circ)            
# net = MQAnsatzOnlyLayer(grad_ops)

# opti = nn.Adam(net.trainable_params(), learning_rate=0.05) 

def train_func(params,pqc):

    loss, gradient = pqc(params)
    print('Time of calc =', time.time() - t1, ' sec')
    print("loss is:",loss)
    return np.real(loss),np.array(np.real(gradient), order="F")


pqc = grad_ops
np.random.seed(2)
inintial_params = []
for i in range(2*p):
    inintial_params.append(random.uniform(0,2 * np.pi)) 


init_params = inintial_params

bound = None
t1 = time.time()
opt = sopt.minimize(train_func, init_params, args=(pqc),
             method='l-BFGS-b',
             jac=True,
             bounds=bound
             )
print(opt.fun)