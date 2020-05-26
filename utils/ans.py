from typing import List, Tuple

import numpy as np
import torch
from torchsearchsorted import searchsorted as torch_searchsorted
# can be installed with sth like: pip install git+https://github.com/aliutkus/torchsearchsorted
# this method should be available in the next pytorch release

class ANS:
    """ANS - bitswap implementation"""
    def __init__(self, pmfs: torch.Tensor, bits: int = 31, quantbits: int = 8):
        self.device = pmfs.device
        self.bits = bits
        self.quantbits = quantbits

        # mask of 2**bits - 1 bits
        self.mask = (1 << bits) - 1

        # normalization constants
        self.lbound = 1 << 32
        self.tail_bits = (1 << 32) - 1

        self.seq_len, self.support = pmfs.shape

        # compute pmf's and cdf's scaled up by 2**n
        multiplier = (1 << self.bits) - (1 << self.quantbits)
        self.pmfs = (pmfs * multiplier).long()

        # add ones to counter zero probabilities
        self.pmfs += torch.ones_like(self.pmfs)

        # add remnant to the maximum value of the probabilites
        self.pmfs[torch.arange(0, self.seq_len), torch.argmax(self.pmfs, dim=1)] += (
                (1 << self.bits) - self.pmfs.sum(1))

        # compute cdf's
        self.cdfs = torch.cumsum(self.pmfs, dim=1)  # compute CDF (scaled up to 2**n)
        self.cdfs = torch.cat([torch.zeros([self.cdfs.shape[0], 1], dtype=torch.long, device=self.device), self.cdfs],
                              dim=1)  # pad with 0 at the beginning

        # move cdf's and pmf's the cpu for faster encoding and decoding
        self.cdfs = self.cdfs.cpu().numpy()
        self.pmfs = self.pmfs.cpu().numpy()

        assert self.cdfs.shape == (self.seq_len, self.support + 1)
        assert np.all(self.cdfs[:, -1] == (1 << bits))

    def encode(self, stream: List[int], sequence: torch.Tensor) -> List[int]:
        for i, s in enumerate(sequence):
            pmf = int(self.pmfs[i, s])
            # print(s, x[-1])
            # print("lbpmf", (((self.lbound >> self.bits) << 32) * pmf))

            if stream[-1] >= ((self.lbound >> self.bits) << 32) * pmf:
                # print(True)
                stream.append(stream[-1] >> 32)
                stream[-2] = stream[-2] & self.tail_bits

            stream[-1] = ((stream[-1] // pmf) << self.bits) + (stream[-1] % pmf) + int(self.cdfs[i, s])

            # print("nst after", x[-1])
            # print("----")
        return stream

    def decode(self, stream: List[int]) -> Tuple[List[int], torch.Tensor]:
        sequence = np.zeros((self.seq_len,), dtype=np.int64)
        for i in reversed(range(self.seq_len)):
            masked_x = stream[-1] & self.mask
            s = np.searchsorted(self.cdfs[i, :-1], masked_x, 'right') - 1
            sequence[i] = s
            stream[-1] = int(self.pmfs[i, s]) * (stream[-1] >> self.bits) + masked_x - int(self.cdfs[i, s])
            if stream[-1] < self.lbound:
                stream[-1] = (stream[-1] << 32) | stream.pop(-2)
        sequence = torch.from_numpy(sequence).to(self.device)
        return stream, sequence

    def batch_encode(self, streams: List[List[int]], sequences: torch.Tensor) -> List[List[int]]:
        assert len(streams) == len(sequences)
        return [
            self.encode(stream, seq)
            for (stream, seq) in zip(streams, sequences)
        ]

    def batch_decode(self, streams: List[List[int]]) -> Tuple[List[List[int]], torch.Tensor]:
        ret_streams = []
        ret_codes = []
        for stream in streams:
            ret_stream, ret_code = self.decode(stream)
            ret_streams.append(ret_stream)
            ret_codes.append(ret_code.cpu().numpy())

        return ret_streams, torch.tensor(ret_codes).long().to(self.device)


class VectorizedANS(ANS):

    def __init__(self, pmfs: torch.Tensor, bits: int = 33, quantbits: int = 8):
        # bits is 33 to keep state in int64 domain
        super().__init__(pmfs, bits, quantbits)
        self.device = pmfs.device
        self.bits = bits
        self.quantbits = quantbits

        # mask of 2**bits - 1 bits
        self.mask = (1 << bits) - 1

        # normalization constants
        self.lbound = 1 << 32
        self.tail_bits = (1 << 32) - 1

        self.seq_len, self.support = pmfs.shape

        # compute pmf's and cdf's scaled up by 2**n
        multiplier = (1 << self.bits) - (1 << self.quantbits)
        self.pmfs = (pmfs * multiplier).long()

        # add ones to counter zero probabilities
        self.pmfs += torch.ones_like(self.pmfs)

        # add remnant to the maximum value of the probabilites
        self.pmfs[torch.arange(0, self.seq_len), torch.argmax(self.pmfs, dim=1)] += (
                (1 << self.bits) - self.pmfs.sum(1))

        # compute cdf's
        self.cdfs = torch.cumsum(self.pmfs, dim=1)  # compute CDF (scaled up to 2**n)
        self.cdfs = torch.cat([torch.zeros([self.cdfs.shape[0], 1], dtype=torch.long, device=self.device), self.cdfs],
                              dim=1)  # pad with 0 at the beginning

        assert self.cdfs.shape == (self.seq_len, self.support + 1)
        assert torch.all(self.cdfs[:, -1] == (1 << bits))

    def encode(self, stream: List[int], sequence: torch.Tensor):
        return self.batch_encode([stream], sequence.unsqueeze(0))[0]

    def decode(self, stream: List[int]):
        return self.batch_decode([stream])[0]

    def batch_encode(self, streams_b_t: List[List[int]], symbols_b_t: torch.Tensor):
        symbols_t_b = symbols_b_t.t()
        for i, s in enumerate(symbols_t_b):
            pmf = self.pmfs[i, s]

            old_streams_tops = torch.tensor(
                [int(s[-1]) for s in streams_b_t],
                device=self.device,
            )
            # print(s, old_streams_tops, old_streams_tops.type())

            rbound = ((self.lbound >> self.bits) << 32)
            # print(rbound, np.log(rbound) / np.log(2))
            overflows = old_streams_tops / pmf >= ((self.lbound >> self.bits) << 32)

            # print("ost/pmf", old_streams_tops / pmf)
            # a hack because multiplying right side by pmf causes overflow, in pytorch we can't go higher than int64

            # print("lbpmf", (((self.lbound >> self.bits) << 32) * pmf))
            new_streams_tops = torch.ones_like(old_streams_tops) * -1
            new_streams_tops[overflows] = old_streams_tops[overflows] >> 32
            old_streams_tops[overflows] = old_streams_tops[overflows] & self.tail_bits
            new_streams_tops[~overflows] = old_streams_tops[~overflows]

            # print("nst", new_streams_tops)
            # print("pmf", pmf)
            # print("nst/pmf", new_streams_tops / pmf)
            # print("nst_pmf_b", ((new_streams_tops / pmf) << self.bits))
            # print("nst%pmf", ((new_streams_tops % pmf)))

            # print("cdf", self.cdfs[i,s])
            new_streams_tops = (
                                       (new_streams_tops // pmf) << self.bits
                               ) + (
                                       new_streams_tops % pmf
                               ) + self.cdfs[i, s]

            # print("nst after", new_streams_tops)
            for (s_b_t, ost, o, n) in zip(streams_b_t, old_streams_tops, overflows, new_streams_tops):
                # print(o.item())
                s_b_t.pop()
                if o.item():
                    # print(ost.item())
                    s_b_t.append(ost.item())

                # print("n", n)
                s_b_t.append(n.item())
                # print("sbt_a", s_b_t[-1])
            #
            # print("sbt", streams_b_t[0][-1])
            # print("---")
        return streams_b_t

    def batch_decode(self, streams_b_t: List[List[int]]) -> Tuple[List[List[int]], torch.Tensor]:
        sequences = [[] for _ in streams_b_t]

        for i in reversed(range(self.seq_len)):
            old_streams_tops = torch.tensor(
                [int(s[-1]) for s in streams_b_t],
                device=self.device,
            )
            older_streams_tops = torch.tensor(
                [int(s[-2]) for s in streams_b_t],
                device=self.device,
            )
            masked_streams_tops = old_streams_tops & self.mask

            symbols = torch_searchsorted(
                self.cdfs[i, :-1].unsqueeze(0), masked_streams_tops.unsqueeze(0), side='right'
            ) - 1

            symbols = symbols.reshape(symbols.shape[1])
            # for seq, s in zip(sequences, symbols):
            #     seq.append(s)

            new_streams_tops = self.pmfs[i, symbols] * (old_streams_tops >> self.bits) + masked_streams_tops - \
                               self.cdfs[i, symbols]
            underflows = new_streams_tops < self.lbound

            new_streams_tops[underflows] = (new_streams_tops[underflows] << 32) | older_streams_tops[underflows]

            for (stream, u, n, seq, s) in zip(streams_b_t, underflows, new_streams_tops, sequences, symbols):
                stream.pop()
                if u.item():
                    stream.pop()
                stream.append(n)
                seq.append(s.item())

            # if streams_b_t[-1] < self.lbound:
            #     streams_b_t[-1] = (streams_b_t[-1] << 32) | streams_b_t.pop(-2)
        # sequences = torch.from_numpy(sequences).to(self.device)
        sequences = torch.tensor([list(reversed(seq)) for seq in sequences]).long().to(self.device)
        return streams_b_t, sequences

if __name__ == '__main__':


    np.random.seed(0)
    base = 16
    total = 2 ** base

    letters = ['A', 'B', 'C']

    probabilities = [0.2, 0.3, 0.5]

    # code_letters_1 = "BABCCCBABCCCAA"#np.random.choice(letters, 4, p=probabilities)
    n_letters = 1000
    n_codes = 100

    codes_letters = []
    codes = []
    states = []

    for i in range(n_codes):
        code_letters = np.random.choice(letters, n_letters, p=probabilities).tolist()
        code = [letters.index(l) for l in code_letters]
        state = list(map(int, np.random.randint(low=1 << 8, high=(1 << 16) - 1, size=10000,
                                              dtype=np.uint32)))  # fill state list with 'random' bits

        codes_letters.append(code_letters)
        codes.append(code)
        states.append(state)

    codes= torch.tensor(codes)

    pmfs = torch.tensor([probabilities for _ in range(codes.shape[1])])


    vans = VectorizedANS(
        pmfs=pmfs,
        bits=33
    )

    state_b_t = states

    from time import time

    s = time()
    state_b_t = vans.batch_encode(state_b_t, codes)
    state_b_t, dec = vans.batch_decode(state_b_t)
    t = time()

    print("vec", t - s)
    assert codes.tolist() == dec.tolist()


    ans = ANS(
        pmfs=pmfs,
        bits=33
    )

    s = time()

    state_b_t = ans.batch_encode(state_b_t, codes)
    state_b_t, dec = ans.batch_decode(state_b_t)
    t = time()

    print("loop", t - s)

    assert codes.tolist() == dec.tolist()

