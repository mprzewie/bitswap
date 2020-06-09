from typing import List, Tuple
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
# we need torch >= 1.6 in order to use searchsorted method. 
# It's not yet officially released, but it can be installed from
# https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html

from time import time
from utils.ans import ANS, VectorizedANS
from copy import deepcopy

harry_potter = '''
Harry turned to look at Ron and Hermione. Neither of them seemed to have
understood what Xenophilius had said either.
"The Deathly Hallows?"
"That's right," said Xenophilius. "You haven't heard of them? I'm not surprised.
Very, very few wizards believe. Witness that knuckle-headed young man at your
brother's wedding," he nodded at Ron, "who attacked me for sporting the symbol of a
well-known Dark wizard! Such ignorance. There is nothing Dark about the Hallows – at
least not in that crude sense. One simply uses the symbol to reveal oneself to other
believers, in the hope that they might help one with the Quest."
He stirred several lumps of sugar into his Gurdyroot infusion and drank some.
"I'm sorry," said Harry, "I still don't really understand."
To be polite, he took a sip from his cup too, and almost gagged: The stuff was
quite disgusting, as though someone had liquidized bogey-flavored Every Flavor Beans. '''


np.random.seed(0)
base = 16
total = 2 ** base

# letters = ['A', 'B', 'C']
# probabilities = [0.2, 0.3, 0.5]

harry_potter = harry_potter.lower()
letters_harry_potter = np.array(list(harry_potter))
unique_letters = np.unique(letters_harry_potter, return_counts=True)

sum_of_all = sum(unique_letters[1])
letter_probabilities = {x: y / sum_of_all for x, y in zip(unique_letters[0], unique_letters[1])}

n_letters = 1000
n_codes = 100
n_repeats = 3

measurements = []

def benchmark_ans(
    ans: ANS,
    state_b_t,
    codes
) -> float:

    state_init = deepcopy(state_b_t)

    s = time()
    state_b_t = ans.batch_encode(state_b_t, codes)
    state_b_t, dec = ans.batch_decode(state_b_t)
    t = time()

    assert codes.tolist() == dec.tolist()
    assert state_b_t == state_init

    return t - s

for n_codes in range (10, 1000, 10):
    codes_letters = []
    codes = []
    states = []


    u_l = unique_letters[0].tolist()



    for i in range(n_codes):
        code_letters = np.pad(letters_harry_potter, pad_width=(0,  1000 - letters_harry_potter.shape[0]), mode='wrap')
        code = [u_l.index(l) for l in code_letters]
        state = list(map(int, np.random.randint(low=1 << 8, high=(1 << 16) - 1, size=10,
                                                dtype=np.uint32)))  # fill state list with 'random' bits

        codes_letters.append(code_letters)
        codes.append(code)
        states.append(state)

    codes = torch.tensor(codes)

    pmfs = torch.tensor(
        [list(letter_probabilities.values()) for _ in range(codes.shape[1])])  # powtarzam to tyle razy jakoa jest długość sekwencji

    vans_cpu = VectorizedANS(
        pmfs=torch.tensor([
            pmfs.numpy()  # tyle razy powtarzam ile mam sekwencji
            for _ in
            range(n_codes)
        ], device=torch.device("cpu")
),
        bits=33
    )

    vans_gpu = VectorizedANS(
        pmfs=torch.tensor([
            pmfs.numpy()  # tyle razy powtarzam ile mam sekwencji
            for _ in
            range(n_codes)
        ], device=torch.device("cuda:0")),
        bits=33
    )

    ans = ANS(
        pmfs=pmfs,
        bits=33
    )
    codes_cpu = codes.to(torch.device("cpu"))
    codes_gpu = codes.to(torch.device("cuda:0"))

    for coder_name, (coder, to_encode) in {
        "ans": (ans, codes_cpu),
        "vans_cpu": (vans_cpu, codes_cpu),
        "vans_gpu": (vans_gpu, codes_gpu)
    }.items():
        for r in range(n_repeats):
            exp_time = benchmark_ans(coder, deepcopy(states), to_encode)
            row = {
                "coder": coder_name,
                "time": exp_time,
                "n_codes": n_codes,
                "repeat": r,
                
            }
            measurements.append(row)
            print(row)

    

data = pd.DataFrame(measurements)
data.to_csv("ans_benchmark.csv")