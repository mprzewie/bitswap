from typing import List, Tuple
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
# we need torch >= 1.6 in order to use searchsorted method. 
# It's not yet officially released, but it can be installed from
# https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html

from time import time

NORM_CONST = 31
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




# 31 bc we need to keep in int domain

class ANS:
    """ANS - bitswap implementation"""

    def __init__(self, pmfs: torch.Tensor, bits: int = 31, quantbits: int = 8):
        init_s = time()
        self.device = pmfs.device
        self.bits = bits
        self.quantbits = quantbits

        # mask of 2**bits - 1 bits
        self.mask = (1 << bits) - 1

        # normalization constants
        self.lbound = 1 << NORM_CONST
        self.tail_bits = (1 << NORM_CONST) - 1

        self.seq_len, self.support = pmfs.shape

        # compute pmf's and cdf's scaled up by 2**n
        t1 = time()

        multiplier = (1 << self.bits) - (1 << self.quantbits)
        self.pmfs = (pmfs * multiplier).long()
        t11 = time()

        # add ones to counter zero probabilities
        self.pmfs += torch.ones_like(self.pmfs)

        t2 = time()
        # print("mult", (t11 - t1), "add", (t2 - t11))

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
        init_e = time()
        # print("ans init", init_e - init_s)

    def encode(self, stream: List[int], sequence: torch.Tensor) -> List[int]:
        for i, s in enumerate(sequence):
            pmf = int(self.pmfs[i, s])
            # print(s, x[-1])
            # print("lbpmf", (((self.lbound >> self.bits) << 32) * pmf))

            if stream[-1] >= ((self.lbound >> self.bits) << NORM_CONST) * pmf:
                # print(True)
                stream.append(stream[-1] >> NORM_CONST)
                stream[-2] = stream[-2] & self.tail_bits

            stream[-1] = ((stream[-1] // pmf) << self.bits) + (stream[-1] % pmf) + int(self.cdfs[i, s])

        return stream

    def decode(self, stream: List[int]) -> Tuple[List[int], torch.Tensor]:
        sequence = np.zeros((self.seq_len,), dtype=np.int64)
        for i in reversed(range(self.seq_len)):
            masked_x = stream[-1] & self.mask
            s = np.searchsorted(self.cdfs[i, :-1], masked_x, 'right') - 1
            sequence[i] = s
            stream[-1] = int(self.pmfs[i, s]) * (stream[-1] >> self.bits) + masked_x - int(self.cdfs[i, s])
            if stream[-1] < self.lbound:
                stream[-1] = (stream[-1] << NORM_CONST) | stream.pop(-2)
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

    def __init__(self, pmfs: torch.Tensor, bits: int = 31, quantbits: int = 8):
        init_s = time()
        if len(pmfs.shape) == 2:
            pmfs = pmfs.unsqueeze(0)

        # super().__init__(pmfs, bits, quantbits)
        self.device = pmfs.device
        self.bits = bits
        self.quantbits = quantbits

        # mask of 2**bits - 1 bits
        self.mask = (1 << bits) - 1

        # normalization constants
        self.lbound = 1 << NORM_CONST
        self.tail_bits = (1 << NORM_CONST) - 1

        self.n_seqs, self.seq_len, self.support = pmfs.shape

        # compute pmf's and cdf's scaled up by 2**n
        multiplier = (1 << self.bits) - (1 << self.quantbits)

        if self.device == torch.device("cpu"):
            self.pmfs = torch.stack([
                (pmf * multiplier) + torch.ones_like(pmf)
                for pmf in pmfs
            ]).long()
        else:
            self.pmfs = ((pmfs * multiplier) + torch.ones_like(pmfs)).long()

        # print("mult", (t11 - t1), "add", (t2 - t11))

        for i in range(self.n_seqs):
            self.pmfs[
                i, torch.arange(0, self.seq_len), torch.argmax(self.pmfs[i], dim=1)
            ] += ((1 << self.bits) - self.pmfs[i].sum(1))
        t3 = time()
        # compute cdf's
        self.cdfs = torch.cumsum(self.pmfs, dim=2)  # compute CDF (scaled up to 2**n)
        # print(self.cdfs.shape)
        self.cdfs = torch.cat(
            [
                torch.zeros([self.n_seqs, self.cdfs.shape[1], 1], dtype=torch.long, device=self.device),
                self.cdfs
            ],
            dim=2)  # pad with 0 at the beginning

        assert self.pmfs.shape == (self.n_seqs, self.seq_len, self.support)
        assert self.cdfs.shape == (self.n_seqs, self.seq_len, self.support + 1)
        assert torch.all(self.cdfs[:, :, -1] == (1 << bits))

        self.pmfs_t_b_p = self.pmfs.transpose(0, 1)
        self.cdfs_t_b_p = self.cdfs.transpose(0, 1)

    def encode(self, stream: List[int], sequence: torch.Tensor):
        return self.batch_encode([stream], sequence.unsqueeze(0))[0]

    def decode(self, stream: List[int]):
        streams, symbols = self.batch_decode([stream])
        return streams[0], symbols[0]

    def batch_encode(self, streams_b_t: List[List[int]], symbols_b_t: torch.Tensor):
        symbols_t_b = symbols_b_t.t()
        old_streams_tops = torch.tensor(
            [int(s[-1]) for s in streams_b_t],
            device=self.device,
        )
        b, t = symbols_b_t.shape

        streams_tensor = torch.ones((b, t + 1)).long().to(self.device) * -1
        h_pointers = torch.zeros(b).long()  # point at tops of the streams
        v_pointers = torch.arange(b).long()
        streams_tensor[v_pointers, h_pointers] = old_streams_tops

        for i, s in tqdm(enumerate(symbols_t_b), desc="batch encode"):
            pmf = self.pmfs_t_b_p[i, torch.arange(self.n_seqs), s]
            cdf = self.cdfs_t_b_p[i, torch.arange(self.n_seqs), s]
            overflows = old_streams_tops / pmf >= ((self.lbound >> self.bits) << NORM_CONST)

            new_streams_tops = torch.ones_like(old_streams_tops) * -1
            new_streams_tops[overflows] = old_streams_tops[overflows] >> NORM_CONST
            old_streams_tops[overflows] = old_streams_tops[overflows] & self.tail_bits
            new_streams_tops[~overflows] = old_streams_tops[~overflows]
            new_streams_tops = (
                                       (new_streams_tops // pmf) << self.bits
                               ) + (
                                       new_streams_tops % pmf
                               ) + cdf

            streams_tensor[overflows, h_pointers[overflows]] = old_streams_tops[overflows]
            h_pointers[overflows] += 1
            streams_tensor[v_pointers, h_pointers] = new_streams_tops
            old_streams_tops = new_streams_tops
        new_streams_b_t = streams_tensor.cpu().tolist()
        h_pointers = h_pointers.cpu().tolist()
        return [
            stream[:-1] + new_stream[:h + 1]
            for stream, new_stream, h in zip(streams_b_t, new_streams_b_t, h_pointers)
        ]

    def batch_decode(self, streams_b_t: List[List[int]]) -> Tuple[List[List[int]], torch.Tensor]:
        sequences = [[] for _ in streams_b_t]
        b = len(streams_b_t)
        v_pointers = torch.arange(b).long()
        h_pointers = torch.tensor(
            [len(s) for s in streams_b_t]
        ).long() - 1  # point at tops of the streams

        max_str_len = h_pointers.max() + 1
        stream_tensor = torch.ones((b, max_str_len)).long() * -1
        for i, stream in enumerate(streams_b_t):
            stream_tensor[i, :len(stream)] = torch.tensor(stream).long()

        stream_tensor = stream_tensor.to(self.device)
        seq_tensor = torch.zeros((self.n_seqs, self.seq_len)).long().to(self.device)

        for i in tqdm(reversed(range(self.seq_len)), desc="batch decode"):
            old_streams_tops = stream_tensor[v_pointers, h_pointers]
            older_streams_tops = stream_tensor[v_pointers, h_pointers - 1]
            masked_streams_tops = old_streams_tops & self.mask
            symbols = torch.searchsorted(
                self.cdfs_t_b_p[i][v_pointers, :-1],
                masked_streams_tops.unsqueeze(1),
                right=True,
            ) - 1

            symbols = symbols.reshape(symbols.shape[0])

            new_streams_tops = self.pmfs_t_b_p[i, v_pointers, symbols] * (
                        old_streams_tops >> self.bits) + masked_streams_tops - self.cdfs_t_b_p[i, v_pointers, symbols]
            underflows = new_streams_tops < self.lbound

            new_streams_tops[underflows] = (new_streams_tops[underflows] << NORM_CONST) | older_streams_tops[underflows]

            stream_tensor[v_pointers, h_pointers] = -1
            h_pointers[underflows] -= 1
            stream_tensor[underflows, h_pointers[underflows]] = -1
            stream_tensor[v_pointers, h_pointers] = new_streams_tops
            seq_tensor[v_pointers, i] = symbols

        stream_tensor = stream_tensor.cpu().tolist()
        h_pointers = h_pointers.cpu().tolist()
        return [
                   stream[:h + 1]
                   for (stream, h) in zip(stream_tensor, h_pointers)
               ], seq_tensor


if __name__ == '__main__':

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

    # code_letters_1 = "BABCCCBABCCCAA"#np.random.choice(letters, 4, p=probabilities)
    n_letters = 1000
    n_codes = 100
    vec_time_array = []
    loop_time_array = []

    for n_codes in range (10, 600, 10):
        codes_letters = []
        codes = []
        states = []


        u_l = unique_letters[0].tolist()



        for i in range(n_codes):
            code_letters = np.pad(letters_harry_potter, pad_width=(0,  1000 - letters_harry_potter.shape[0]), mode='wrap')
            code = [u_l.index(l) for l in code_letters]
            state = list(map(int, np.random.randint(low=1 << 8, high=(1 << 16) - 1, size=10000,
                                                    dtype=np.uint32)))  # fill state list with 'random' bits

            codes_letters.append(code_letters)
            codes.append(code)
            states.append(state)


        codes = torch.tensor(codes)

        pmfs = torch.tensor(
            [list(letter_probabilities.values()) for _ in range(codes.shape[1])])  # powtarzam to tyle razy jakoa jest długość sekwencji

        vans = VectorizedANS(
            pmfs=torch.tensor([
                pmfs.numpy()  # tyle razy powtarzam ile mam sekwencji
                for _ in
                range(n_codes)
            ]),
            bits=33
        )

        state_b_t = states

        from time import time

        s = time()
        state_b_t = vans.batch_encode(state_b_t, codes)
        state_b_t, dec = vans.batch_decode(state_b_t)
        t = time()

        print("vec", t - s)

        vec_time_array.append(t - s)
        assert codes.tolist() == dec.tolist()

        ans = ANS(
            pmfs=pmfs,
            bits=33
        )

        s = time()


        state_b_t = ans.batch_encode(state_b_t, codes)
        state_b_t, dec = ans.batch_decode(state_b_t)
        t = time()

        loop_time_array.append(t - s)

        print("loop", t - s)
        assert codes.tolist() == dec.tolist()

    data = pd.DataFrame({"vec" : vec_time_array, "loop": loop_time_array })
    data.to_csv("data1.csv")