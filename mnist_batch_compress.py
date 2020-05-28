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
    device = "cpu" #f"cuda:{gpu}" # gpu

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
    experiments = 10
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
        state = list(map(int, np.random.randint(low=1 << 16, high=(1 << NORM_CONST) - 1, size=10000, dtype=np.uint32))) # fill state list with 'random' bits
        state[-1] = state[-1] << 16 #NORM_CONST
        
        states = [
            state.copy()
            for _ in range(len(datapoints))
        ]

        initialstates = deepcopy(states)
        reststates = None

        
        iterator = tqdm(range(len(datapoints)), desc="Sender")

        # <===== SENDER =====>

        pmfs_cache = dict()
        ####
        for zi in range(nz):
            z_enc_pmfs = []
            xs = []
            for xi in tqdm(range(len(datapoints))):
                (x, _) = datapoints[xi]
                x = x.to(device).view(xdim)
                

                input = zcentres[zi - 1, zrange, zsyms[xi]] if zi > 0 else xcentres[xrange, x.long()]
                mu, scale = model.infer(zi)(given=input)
                cdfs = logistic_cdf(zendpoints[zi].t(), mu, scale).t() # most expensive calculation?
                pmfs = cdfs[:, 1:] - cdfs[:, :-1]
                pmfs = torch.cat((cdfs[:, 0].unsqueeze(1), pmfs, 1. - cdfs[:, -1].unsqueeze(1)), dim=1)

                z_enc_pmfs.append(pmfs)
                xs.append(x)
            
            ans = ANS(
                torch.stack(z_enc_pmfs),
                bits=ansbits, quantbits=quantbits
            )

            states, zsymtops = ans.batch_decode(states)

            if zi == 0:
                reststates = states.copy()
                assert all([
                    len(rb) > 1
                    for rb in reststates
                ]),  "too few initial bits" # otherwise initial state consists of too few bits

            z_dec_pmfs = []
            for zsymtop in tqdm(zsymtops):
                z = zcentres[zi, zrange, zsymtop]
                mu, scale = model.generate(zi)(given=z)
                cdfs = logistic_cdf((zendpoints[zi - 1] if zi > 0 else xendpoints).t(), mu, scale).t() # most expensive calculation?
                pmfs = cdfs[:, 1:] - cdfs[:, :-1]
                pmfs = torch.cat((cdfs[:, 0].unsqueeze(1), pmfs, 1. - cdfs[:, -1].unsqueeze(1)), dim=1)

                z_dec_pmfs.append(pmfs)
            
            ans = ANS(
                torch.stack(z_dec_pmfs),
                bits=ansbits, quantbits=quantbits
            )

            to_encode = zsyms if zi > 0 else torch.stack(xs).long()
            states = ans.batch_encode(
                states,
                to_encode
            )

            zsyms = zsymtops


        ###

        states = ANS(
            prior_pmfs, bits=ansbits, quantbits=quantbits
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
        for xi, (totaladdedbits, totalbits) in iterator:
            (x, _) = datapoints[xi]
            x = x.to(device).view(xdim)
            with torch.no_grad():
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
        with open(state_file, "wb") as fp:
            pickle.dump(states, fp)

        state = None
        # open state file
        with open(state_file, "rb") as fp:
            states = pickle.load(fp)

        # <===== RECEIVER =====>


        # priors
        states, zsymtops = ANS(
            prior_pmfs, bits=ansbits, quantbits=quantbits
        ).batch_decode(states)

        for zi in reversed(range(nz)):
            zs = z = zcentres[zi, zrange, zsymtops]

            z_dec_pmfs = []
            for xi in tqdm(range(len(datapoints))):

                z = zs[xi]
                mu, scale = model.generate(zi)(given=z)
                cdfs = logistic_cdf((zendpoints[zi - 1] if zi > 0 else xendpoints).t(), mu, scale).t()  # most expensive calculation?
                pmfs = cdfs[:, 1:] - cdfs[:, :-1]
                pmfs = torch.cat((cdfs[:, 0].unsqueeze(1), pmfs, 1. - cdfs[:, -1].unsqueeze(1)), dim=1)
                z_dec_pmfs.append(pmfs)
            
            ans = ANS(
                torch.stack(z_dec_pmfs),
                bits=ansbits, quantbits=quantbits
            )
            
            states, symbols = ans.batch_decode(states)

            inputs = zcentres[zi - 1, zrange, symbols] if zi > 0 else xcentres[xrange, symbols]

            z_enc_pmfs = []
            for input in tqdm(inputs):
                mu, scale = model.infer(zi)(given=input)
                cdfs = logistic_cdf(zendpoints[zi].t(), mu, scale).t() # most expensive calculation?
                pmfs = cdfs[:, 1:] - cdfs[:, :-1]
                pmfs = torch.cat((cdfs[:, 0].unsqueeze(1), pmfs, 1. - cdfs[:, -1].unsqueeze(1)), dim=1)
                z_enc_pmfs.append(pmfs)

            ans = ANS(
                torch.stack(z_enc_pmfs),
                bits=ansbits, quantbits=quantbits
            )

            states = ans.batch_encode(states, zsymtops)

            zsymtops = symbols

        assert all([
            torch.all(datapoints[xi][0].view(xdim).long().to(device) == zsymtops[xi])
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