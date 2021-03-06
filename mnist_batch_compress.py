from utils.torch.rand import *
from utils.torch.modules import ImageNet
from model.mnist_train import Model
from torch.utils.data import *
from discretization import *
from torchvision import datasets, transforms
import random
import time
import argparse
from tqdm import tqdm
import pickle
from utils.ans import NORM_CONST, ANS, VectorizedANS as ANS
from copy import deepcopy

def compress(quantbits, nz, bitswap, gpu):
    # model and compression params
    zdim = 1 * 16 * 16
    zrange = torch.arange(zdim)
    xdim = 32 ** 2 * 1
    xrange = torch.arange(xdim)
    ansbits = NORM_CONST - 1 # ANS precision
    type = torch.float64 # datatype throughout compression
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ans_device = device #"cuda:0"

    # set up the different channel dimension for different latent depths
    if nz == 8:
        reswidth = 61
    elif nz == 4:
        reswidth = 62
    elif nz == 2:
        reswidth = 63
    else:
        reswidth = 64
    assert nz > 0

    print(f"{'Bit-Swap' if bitswap else 'BB-ANS'} - MNIST - {nz} latent layers - {quantbits} bits quantization")

    # seed for replicating experiment and stability
    np.random.seed(100)
    random.seed(50)
    torch.manual_seed(50)
    torch.cuda.manual_seed(50)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # compression experiment params
    experiments = 20
    ndatapoints = 100
    decompress = True

    # <=== MODEL ===>
    model = Model(
        xs = (1, 32, 32), nz=nz, zchannels=1, 
        nprocessing=4, kernel_size=3, resdepth=8, 
        reswidth=reswidth,
        tag="batch"
        ).to(device)
    model.load_state_dict(
        torch.load(f'model/params/mnist/nz{nz}',
                   map_location=lambda storage, location: storage
                   )
    )
    model.eval()

    print("Discretizing")
    # get discretization bins for latent variables
    zendpoints, zcentres = discretize(nz, quantbits, type, device, model, "mnist")

    #### priors
    prior_cdfs = logistic_cdf(zendpoints[-1].t(), torch.zeros(1, device=device, dtype=type), torch.ones(1, device=device, dtype=type)).t()
    prior_pmfs = prior_cdfs[:, 1:] - prior_cdfs[:, :-1]
    prior_pmfs = torch.cat((prior_cdfs[:, 0].unsqueeze(1), prior_pmfs, 1. - prior_cdfs[:, -1].unsqueeze(1)), dim=1)

    ####

    # get discretization bins for discretized logistic
    xbins = ImageBins(type, device, xdim)
    xendpoints = xbins.endpoints()
    xcentres = xbins.centres()

    print("Load data..")
    # <=== DATA ===>
    class ToInt:
        def __call__(self, pic):
            return pic * 255
    transform_ops = transforms.Compose([transforms.Pad(2), transforms.ToTensor(), ToInt()])
    test_set = datasets.MNIST(root="model/data/mnist", train=False, transform=transform_ops, download=True)

    # sample (experiments, ndatapoints) from test set with replacement
    print(len(test_set.data))
    if not os.path.exists("bitstreams/mnist/indices"):
        randindices = np.random.choice(len(test_set.data), size=(experiments, ndatapoints), replace=False)
        np.save("bitstreams/mnist/indices", randindices)
    else:
        randindices = np.load("bitstreams/mnist/indices")

    print("Setting up metrics..")
    # metrics for the results
    nets = np.zeros((experiments, ndatapoints), dtype=np.float)
    elbos = np.zeros((experiments, ndatapoints), dtype=np.float)
    cma = np.zeros((experiments, ndatapoints), dtype=np.float)
    total = np.zeros((experiments, ndatapoints), dtype=np.float)

    print("Compression..")
    for ei in range(experiments):
        experiment_start_time = time.time()
        print(f"Experiment {ei + 1}")
        subset = Subset(test_set, randindices[ei])
        test_loader = DataLoader(
            dataset=subset,
            batch_size=1, shuffle=False, drop_last=True)
        datapoints = list(test_loader)

        # < ===== COMPRESSION ===>
        # initialize compression
        model.compress()
        state = list(map(int, np.random.randint(low=1 << 16, high=(1 << NORM_CONST) - 1, size=(200), dtype=np.uint32))) # fill state list with 'random' bits
        state[-1] = state[-1] << 16 #NORM_CONST
        
        states = [
            state.copy()
            for _ in range(len(datapoints))
        ]

        initialstates = deepcopy(states)
        reststates = None

        state_init = time.time()

        
        iterator = tqdm(range(len(datapoints)), desc="Sender")

        # <===== SENDER =====>

        ####
        xs = []
        for xi in range(len(datapoints)):
            (x, _) = datapoints[xi]
            x = x.to(device).view(xdim)
            xs.append(x)

        for zi in range(nz):
            mus = []
            scales = []
            for xi in tqdm(range(len(datapoints))):
                input = zcentres[zi - 1, zrange, zsyms[xi]] if zi > 0 else xcentres[xrange, xs[xi].long()]
                mu, scale = model.infer(zi)(given=input)
                mus.append(mu)
                scales.append(scale)


            s = time.time()
            cdfs_b = logistic_cdf(
                torch.stack(
                    [zendpoints[zi]]*len(datapoints)
                    ).permute(2, 0, 1), 
                torch.stack(mus), 
                torch.stack(scales)
            ).permute(1, 2, 0)

            pmfs_b = torch.cat((
                cdfs_b[:, :, 0].unsqueeze(2), 
                cdfs_b[:, :, 1:] - cdfs_b[:, :, :-1], 
                1. - cdfs_b[:, :, -1].unsqueeze(2)
            ), dim=2)
                
            ans = ANS(
                pmfs_b.to(ans_device),
                bits=ansbits, quantbits=quantbits
            )
            t1 = time.time()
            states, zsymtops = ans.batch_decode(states)
            t2 = time.time()
            zsymtops = zsymtops.to(device)

            if zi == 0:
                reststates = states.copy()
                assert all([
                    len(rb) > 1
                    for rb in reststates
                ]),  "too few initial bits" # otherwise initial state consists of too few bits

            z_dec_pmfs = []
            mus = []
            scales = []
            for zsymtop in tqdm(zsymtops):
                z = zcentres[zi, zrange, zsymtop]
                mu, scale = model.generate(zi)(given=z)
                mus.append(mu)
                scales.append(scale)
            
            cdfs_b = logistic_cdf(
                torch.stack(
                    [
                        (zendpoints[zi - 1] if zi > 0 else xendpoints)
                    ]*len(datapoints)
                ).permute(2, 0, 1), 
                torch.stack(mus), 
                torch.stack(scales)
            ).permute(1, 2, 0)

            pmfs_b = torch.cat((
                cdfs_b[:, :, 0].unsqueeze(2), 
                cdfs_b[:, :, 1:] - cdfs_b[:, :, :-1], 
                1. - cdfs_b[:, :, -1].unsqueeze(2)
            ), dim=2)
            
            ans = ANS(
                pmfs_b.to(ans_device),
                bits=ansbits, quantbits=quantbits
            )

            to_encode = zsyms if zi > 0 else torch.stack(xs).long()
            states = ans.batch_encode(
                states,
                to_encode
            )

            zsyms = zsymtops

        states = ANS(
            torch.stack([
                prior_pmfs
                for _ in range(len(datapoints))
            ]).to(ans_device), 
            bits=ansbits, quantbits=quantbits
        ).batch_encode(states, zsymtops)
        

        totaladdedbits_for_xs = [
            (len(state) - len(initialstate)) * 32
            for (state, initialstate)
            in zip(states, initialstates)
        ]

        totalbits_for_xs = [
            (len(state) - (len(restbits) - 1)) * 32
            for (state, restbits)
            in zip(states, reststates)
        ]

        iterator = tqdm(
            enumerate(
            zip(totaladdedbits_for_xs, totalbits_for_xs)
        ))
        with torch.no_grad():
            for xi, (totaladdedbits, totalbits) in iterator:
                x = xs[xi]
                model.compress(False)
                logrecon, logdec, logenc, _ = model.loss(x.view((-1,) + model.xs))
                elbo = -logrecon + torch.sum(-logdec + logenc)
                model.compress(True)

                nets[ei, xi] = (totaladdedbits / xdim) - nets[ei, :xi].sum()
                elbos[ei, xi] = elbo.item() / xdim
                cma[ei, xi] = totalbits / (xdim * (xi + 1))
                total[ei, xi] = totalbits

                iterator.set_postfix_str(s=f"N:{nets[ei,:xi+1].mean():.2f}±{nets[ei,:xi+1].std():.2f}, D:{nets[ei,:xi+1].mean()-elbos[ei,:xi+1].mean():.4f}, C: {cma[ei,:xi+1].mean():.2f}, T: {totalbits:.0f}", refresh=False)


        state_file = f"bitstreams/mnist/nz{nz}/{'Bit-Swap' if bitswap else 'BB-ANS'}/{'Bit-Swap' if bitswap else 'BB-ANS'}_{quantbits}bits_nz{nz}_experiment{ei + 1}_batch"
        print(state_file)
        # write state to file
        # print(len(states))
        # print([len(s) for s in states])

        max_common_len = min([len(s) for s in states])
        common_len = 0
    
        for pref in range(max_common_len):
            if len(set(s[pref] for s in states)) > 1:
                break
            common_len = pref + 1

        print("common len:", common_len)
        states_to_dump = (
            states[0][:common_len],
            [
                s[common_len:]
                for s in states
            ]
        )
        with open(state_file, "wb") as fp:
            pickle.dump(states_to_dump, fp)

        state = None
        # open state file
        with open(state_file, "rb") as fp:
            states_prefix, states_postfixes = pickle.load(fp)
            states = [
                states_prefix + sp 
                for sp in states_postfixes
            ]
        
        print([len(s) for s in states])
        print(sum([
            len(s) - len(inits)
            for (s, inits) in zip(states, initialstates)
        ]))

        # <===== RECEIVER =====>

        # priors
        states, zsymtops = ANS(
            torch.stack([
                prior_pmfs
                for _ in range(len(datapoints))
            ]).to(ans_device), 
            bits=ansbits, quantbits=quantbits
        ).batch_decode(states)
        zsymtops = zsymtops.to(device)

        for zi in reversed(range(nz)):
            zs = z = zcentres[zi, zrange, zsymtops]

            z_dec_pmfs = []
            mus = []
            scales = []
            for xi in tqdm(range(len(datapoints))):

                z = zs[xi]
                mu, scale = model.generate(zi)(given=z)
                mus.append(mu)
                scales.append(scale)
            
            cdfs_b = logistic_cdf(
                torch.stack(
                    [(zendpoints[zi - 1] if zi > 0 else xendpoints)]*len(datapoints)
                    ).permute(2, 0, 1), 
                torch.stack(mus), 
                torch.stack(scales)
            ).permute(1, 2, 0)

            pmfs_b = torch.cat((
                cdfs_b[:, :, 0].unsqueeze(2), 
                cdfs_b[:, :, 1:] - cdfs_b[:, :, :-1], 
                1. - cdfs_b[:, :, -1].unsqueeze(2)
            ), dim=2)
            
            ans = ANS(
                pmfs_b.to(ans_device),
                bits=ansbits, quantbits=quantbits
            )
            
            states, symbols = ans.batch_decode(states)
            symbols = symbols.to(device)

            inputs = zcentres[zi - 1, zrange, symbols] if zi > 0 else xcentres[xrange, symbols]

            mus = []
            scales = []

            for input in tqdm(inputs):
                mu, scale = model.infer(zi)(given=input)
                mus.append(mu)
                scales.append(scale)

            cdfs_b = logistic_cdf(
                torch.stack(
                    [zendpoints[zi]]*len(datapoints)
                    ).permute(2, 0, 1), 
                torch.stack(mus), 
                torch.stack(scales)
            ).permute(1, 2, 0)

            pmfs_b = torch.cat((
                cdfs_b[:, :, 0].unsqueeze(2), 
                cdfs_b[:, :, 1:] - cdfs_b[:, :, :-1], 
                1. - cdfs_b[:, :, -1].unsqueeze(2)
            ), dim=2)

            ans = ANS(
                pmfs_b.to(ans_device),
                bits=ansbits, quantbits=quantbits
            )

            states = ans.batch_encode(states, zsymtops)
            zsymtops = symbols

        assert all([
            torch.all(datapoints[xi][0].view(xdim).long().to(device) == zsymtops[xi].to(device))
            for xi in range(len(datapoints))
        ])

        assert initialstates == states
        experiment_end_time = time.time()
        print("Experiment time", experiment_end_time - experiment_start_time)

    print(f"N:{nets.mean():.4f}±{nets.std():.2f}, E:{elbos.mean():.4f}±{elbos.std():.2f}, D:{nets.mean() - elbos.mean():.6f}")

    # save experiments
    np.save(f"plots/mnist{nz}/{'bitswap' if bitswap else 'bbans'}_{quantbits}bits_nets",nets)
    np.save(f"plots/mnist{nz}/{'bitswap' if bitswap else 'bbans'}_{quantbits}bits_elbos", elbos)
    np.save(f"plots/mnist{nz}/{'bitswap' if bitswap else 'bbans'}_{quantbits}bits_cmas",cma)
    np.save(f"plots/mnist{nz}/{'bitswap' if bitswap else 'bbans'}_{quantbits}bits_total", total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)  # assign to gpu
    parser.add_argument('--nz', default=2, type=int)  # choose number of latent variables
    parser.add_argument('--quantbits', default=10, type=int)  # choose discretization precision
    parser.add_argument('--bitswap', default=1, type=int)  # choose whether to use Bit-Swap or not

    args = parser.parse_args()
    print(args)

    gpu = args.gpu
    nz = args.nz
    quantbits = args.quantbits
    bitswap = args.bitswap

    for nz in [nz]:
        for bits in [quantbits]:
            for bitswap in [bitswap]:
                compress(bits, nz, bitswap, gpu)