import dataclasses
import math
import time
from typing import Callable, Dict, List, Union, Optional, Any

import numpy as np
import torch
import torch.distributions
import torch.nn.functional as F
import transformers


@dataclasses.dataclass
class History:
    ids: List = dataclasses.field(default_factory=lambda: [])
    xentropy: List = dataclasses.field(default_factory=lambda: [])
    target: List = dataclasses.field(default_factory=lambda: [])
    sim: List = dataclasses.field(default_factory=lambda: [])
    keep: List = dataclasses.field(default_factory=lambda: [])

    def insert(self, new_ids, target, xentropy, sim, keep):
        self.ids.append(new_ids.detach().cpu().numpy())
        self.target.append(target.detach().cpu().numpy())
        self.xentropy.append(xentropy.detach().cpu().numpy())
        self.sim.append(sim.detach().cpu().numpy())
        self.keep.append(keep.detach().cpu().numpy())

    def finalize(self):
        self.ids = np.stack(self.ids, axis=0)
        self.target = np.stack(self.target, axis=0)
        self.xentropy = np.stack(self.xentropy, axis=0)
        self.sim = np.stack(self.sim, axis=0)
        self.keep = np.stack(self.keep, axis=0)

def cat_if_not_none(a, b):
    if a is None or b is None:
        return None
    else:
        return torch.cat((a, b), dim=0)


@dataclasses.dataclass
class State:
    ids: torch.Tensor
    target: torch.Tensor
    xentropy: torch.Tensor
    sim: torch.Tensor
    final_token: torch.Tensor
    token_grads: torch.Tensor
    extra: Dict[str, torch.Tensor]

    def cat(self, state2):
        return State(
            ids=torch.cat((self.ids, state2.ids), dim=0),
            target=torch.cat((self.target, state2.target), dim=0),
            xentropy=torch.cat((self.xentropy, state2.xentropy), dim=0),
            sim=torch.cat((self.sim, state2.sim), dim=0),
            final_token=torch.cat((self.final_token, state2.final_token), dim=0),
            token_grads=cat_if_not_none(self.token_grads, state2.token_grads),
            extra={
                k: cat_if_not_none(self.extra[k], state2.extra[k]) for k in self.extra
            },
        )

    def subset(self, keep):
        return State(
            ids=self.ids[keep],
            target=self.target[keep],
            xentropy=self.xentropy[keep],
            sim=self.sim[keep],
            final_token=self.final_token[keep],
            token_grads=self.token_grads[keep]
            if self.token_grads is not None
            else None,
            extra={k: self.extra[k][keep] for k in self.extra},
        )


def default_callback(tokenizer, nprint=2):
    def f(i, state, last_runtime, history):
        def report_example(j):
            xentropy = state.xentropy[j]
            target = state.target[j]
            sim = state.sim[j]
            current_str = tokenizer.decode(state.ids[j].tolist())
            next_token = tokenizer.decode([state.final_token[j]])
            full_str = current_str + "[" + next_token + "]"
            print(f"  {xentropy=:.2f} {target=:.2f} {sim=:.2f} {repr(full_str)}")
            print(f"  {state.ids[j.tolist()]}")

        if last_runtime is not None:
            print("runtime: {:.2f} seconds".format(last_runtime))
        print(f"\nbeginning step {i}")
        print("lowest xentropy:")
        _nprint = min(nprint, state.ids.shape[0])
        lowest_xentropy = (-state.xentropy).topk(k=_nprint).indices
        for j in range(_nprint):
            report_example(lowest_xentropy[j])

        print("highest target")
        highest_target = state.target.topk(k=_nprint).indices
        for j in range(_nprint):
            report_example(highest_target[j])
        return False

    return f


def eval_xentropy(cache, input_ids):
    batch_size, seq_len, vocab_size = cache["logits"].shape
    loss_fnc = torch.nn.CrossEntropyLoss(reduction="none")
    return (
        loss_fnc(
            cache["logits"][:, :-1].reshape(-1, vocab_size),
            input_ids[:, 1:].reshape(-1),
        )
        .view(batch_size, seq_len - 1)
        .mean(dim=-1)
    )


def pad_to_pow2(n):
    if n < 8:  # no need to go below 8...
        return 8
    return 2 ** math.ceil(math.log2(n))


# based on https://github.com/llm-attacks/llm-attacks/blob/main/llm_attacks/gcg/gcg_attack.py
def token_grads(
    model: torch.nn.Module,
    cache_run: Callable,
    input_ids: torch.Tensor,
    x_penalty: torch.Tensor,
    sim_penalty: float,
    batch_size: int,
):
    """
    Compute gradients with respect to one-hot encoded input tokens. This is a
    infinitesimal approximation to the token influence on the loss so it's a
    very noisy indicator of which tokens might reduce loss.
    """
    embed = model.get_input_embeddings()

    token_grads = torch.empty(
        (input_ids.shape[0], input_ids.shape[1], embed.num_embeddings),
        dtype=torch.float,
        device=model.device,
    )
    loss = torch.empty(input_ids.shape[0], device=model.device)
    sim = torch.empty(input_ids.shape[0], device=model.device)
    xentropy = torch.empty(input_ids.shape[0], device=model.device)
    target = torch.empty(input_ids.shape[0], device=model.device)
    final_token = torch.empty(input_ids.shape[0], device=model.device, dtype=torch.long)
    extra = dict()

    with torch.enable_grad():
        model.zero_grad()

        for i in range(0, input_ids.shape[0], batch_size):
            imax = min(i + batch_size, input_ids.shape[0])

            # using a one hot matrix as input to the model gives us gradients with
            # respect to potential input tokens.
            one_hot = F.one_hot(
                input_ids[i:imax].clone(), num_classes=embed.num_embeddings
            ).type(embed.weight.dtype)
            one_hot.requires_grad = True
            inputs_embeds = torch.matmul(one_hot, embed.weight)

            bs = imax - i
            # pad to next power of two so that we don't compile the model too
            # many times
            # NOTE: PADDING IS OFF!!
            pad = 0  # pad_to_pow2(bs) - bs if bs < batch_size else 0
            # pad = batch_size - bs
            padded_inputs_embeds = torch.cat(
                (
                    inputs_embeds,
                    torch.zeros(
                        (
                            pad,
                            inputs_embeds.shape[1],
                            inputs_embeds.shape[2],
                        ),
                        dtype=inputs_embeds.dtype,
                        device=inputs_embeds.device,
                    ),
                ),
                dim=0,
            )
            cache = cache_run(inputs_embeds=padded_inputs_embeds)

            this_xentropy = (
                -(torch.log_softmax(cache["logits"][:bs, :-1], dim=-1) * one_hot[:, 1:])
                .sum(dim=-1)
                .mean(dim=-1)
            )

            this_loss = (
                -cache["target"][:bs]
                + this_xentropy * x_penalty[i:imax]
                + sim_penalty * cache["sim"][:bs]
            )
            this_loss.sum().backward()

            loss[i:imax] = this_loss
            target[i:imax] = cache["target"][:bs]
            sim[i:imax] = cache["sim"][:bs]
            xentropy[i:imax] = this_xentropy
            final_token[i:imax] = cache["logits"][:bs, -1, :].argmax(dim=-1)
            token_grads[i:imax] = one_hot.grad.detach()#to("cpu")

            for k in cache:
                if k not in ["target", "sim", "logits"]:
                    e = cache[k]
                    if k not in extra:
                        extra[k] = torch.empty(
                            (input_ids.shape[0], *e.shape[1:]),
                            dtype=e.dtype,
                            device=e.device,
                        )
                    extra[k][i:imax] = e[:bs]

            # important to zero out gradients here to release memory
            model.zero_grad()

    return State(input_ids, target, xentropy, sim, final_token, token_grads, extra)


def evaluate_fitness(
    model: torch.nn.Module,
    cache_run: Callable,
    input_ids: torch.Tensor,
    batch_size: int,
):
    target = torch.empty(input_ids.shape[0], dtype=torch.float, device=input_ids.device)
    xentropy = torch.empty(
        input_ids.shape[0], dtype=torch.float, device=input_ids.device
    )
    sim = torch.empty(input_ids.shape[0], dtype=torch.float, device=input_ids.device)
    final_token = torch.empty(
        input_ids.shape[0], dtype=torch.long, device=input_ids.device
    )
    extra = dict()
    for i in range(0, input_ids.shape[0], batch_size):
        imax = min(i + batch_size, input_ids.shape[0])
        bs = imax - i
        # pad to next power of two so that we don't compile the model too
        # many times
        # pad = (2 ** math.ceil(math.log2(bs))) - bs if bs < batch_size else 0
        # NOTE: PADDING IS OFF!!
        pad = 0
        # pad = batch_size - bs
        padded_input_ids = torch.cat(
            (
                input_ids[i:imax],
                torch.zeros(
                    (pad, input_ids.shape[1]),
                    dtype=torch.long,
                    device=input_ids.device,
                ),
            ),
            dim=0,
        )
        mini_batch = cache_run(input_ids=padded_input_ids[:bs])
        target[i:imax] = mini_batch["target"][:bs]
        xentropy[i:imax] = eval_xentropy(
            mini_batch,
            input_ids[i:imax],
        )[:bs]
        sim[i:imax] = mini_batch["sim"][:bs]
        final_token[i:imax] = mini_batch["logits"][:bs, -1, :].argmax(dim=-1)

        for k in mini_batch:
            if k not in ["target", "sim", "logits"]:
                e = mini_batch[k]
                if k not in extra:
                    extra[k] = torch.empty(
                        (input_ids.shape[0], *e.shape[1:]),
                        dtype=e.dtype,
                        device=e.device,
                    )
                extra[k][i:imax] = e[:bs]

    return State(input_ids, target, xentropy, sim, final_token, None, extra)


class Selector:
    def __init__(
        self,
        model: torch.nn.Module,
        cache_run: Callable,
        X: torch.Tensor,
        sim_penalty: float,
        batch_size: int
    ):
        self.model = model
        self.cache_run = cache_run
        self.X = X
        self.sim_penalty = sim_penalty
        self.batch_size = batch_size


class GradientSelector(Selector):
    uses_gradient = True

    def setup(self, input_ids: torch.Tensor):
        return token_grads(
            self.model,
            self.cache_run,
            input_ids,
            x_penalty=self.X[: input_ids.shape[0]],
            sim_penalty=self.sim_penalty,
            batch_size=self.batch_size,
        )

    def mutate(self, state, source_idx, input_ids, topk):
        # when just flipping, the current token gradient falls out of the
        # topk operation, so we can just use the negative new token grad
        topk_grad = (-state.token_grads).topk(k=topk, dim=-1)
        pos = torch.randint(
            low=0,
            high=input_ids.shape[1],
            size=(input_ids.shape[0],),
            device=input_ids.device,
        )
        token_idx = torch.randint(
            low=0,
            high=topk,
            size=(input_ids.shape[0],),
            device=input_ids.device,
        )
        input_ids[torch.arange(input_ids.shape[0]), pos] = topk_grad.indices.to(
            input_ids.device
        )[source_idx, pos, token_idx]

class GradientTokenMaskedSelector(GradientSelector):
    def __init__(self, model, cache_run: Callable[..., Any], X: torch.Tensor, sim_penalty: float, batch_size: int, token_mask: torch.Tensor):
        super().__init__(model, cache_run, X, sim_penalty, batch_size)
        self.token_mask = token_mask
        assert token_mask.shape[0] == model.get_input_embeddings().num_embeddings, f"{token_mask.shape=} {model.get_input_embeddings().num_embeddings=}"
        assert token_mask.ndim == 1, f"{token_mask.ndim=}"

    def mutate(self, state: State, source_ids, input_ids, topk):
        state.token_grads = torch.where(self.token_mask[None, None, :], state.token_grads, torch.finfo(state.token_grads.dtype).max)
        super().mutate(state, source_ids, input_ids, topk)


class GradientCrossoverSelector(Selector):
    uses_gradient = True

    def setup(self, input_ids: torch.Tensor):
        return token_grads(
            self.model,
            self.cache_run,
            input_ids,
            x_penalty=self.X[: input_ids.shape[0]],
            sim_penalty=self.sim_penalty,
            batch_size=self.batch_size,
        )

    def mutate(self, state, source_idx, input_ids, topk):
        # when just flipping, the current token gradient falls out of the
        # topk operation, so we can just use the negative new token grad
        topk_grad = (-state.token_grads).to(input_ids.device).topk(k=topk, dim=-1)
        pos = torch.randint(
            low=0,
            high=input_ids.shape[1],
            size=(input_ids.shape[0],),
            device=input_ids.device,
        )
        token_idx = torch.randint(
            low=0,
            high=topk,
            size=(input_ids.shape[0],),
            device=input_ids.device,
        )
        input_ids[torch.arange(input_ids.shape[0]), pos] = \
            topk_grad.indices[
            source_idx, pos, token_idx
        ]

        nx = input_ids.shape[0] // 2
        xi = torch.randint(
            low=0,
            high=input_ids.shape[0],
            size=(nx,),
        )
        xj = torch.randint(
            low=0,
            high=input_ids.shape[0],
            size=(nx,),
        )
        xsplit = torch.randint(
            low=1,
            high=input_ids.shape[1] - 1,
            size=(nx,),
        )
        for k in range(nx):
            input_ids[xi[k], xsplit[k] :] = input_ids[xj[k], xsplit[k] :]


class GradientMoveSelector(Selector):
    uses_gradient = True

    def setup(self, input_ids: torch.Tensor):
        return token_grads(
            self.model,
            self.cache_run,
            input_ids,
            x_penalty=self.X[: input_ids.shape[0]],
            sim_penalty=self.sim_penalty,
            batch_size=self.batch_size,
        )

    def mutate(self, state, source_idx, input_ids, topk):
        # when just flipping, the current token gradient falls out of the
        # topk operation, so we can just use the negative new token grad
        topk_grad = (-state.token_grads).to(input_ids.device).topk(k=topk, dim=-1)
        pos = torch.randint(
            low=0,
            high=input_ids.shape[1],
            size=(input_ids.shape[0],),
            device=input_ids.device,
        )
        token_idx = torch.randint(
            low=0,
            high=topk,
            size=(input_ids.shape[0],),
            device=input_ids.device,
        )
        input_ids[torch.arange(input_ids.shape[0]), pos] = topk_grad.indices[
            source_idx, pos, token_idx
        ]

        nm = 2
        mi = torch.randint(
            low=0,
            high=input_ids.shape[0],
            size=(nm,),
        )
        mpos = torch.randint(
            low=0,
            high=input_ids.shape[1],
            size=(nm,),
        )
        temp = input_ids[mi, -1].clone()
        input_ids[mi, -1] = input_ids[mi, mpos]
        input_ids[mi, mpos] = temp


class GlobalGradientSelector(Selector):
    uses_gradient = True

    def setup(self, input_ids: torch.Tensor):
        return token_grads(
            self.model,
            self.cache_run,
            input_ids,
            x_penalty=self.X[: input_ids.shape[0]],
            sim_penalty=self.sim_penalty,
            batch_size=self.batch_size,
        )

    def mutate(self, state, source_idx, input_ids, topk):
        cur_grad = state.token_grads[
            torch.arange(state.ids.shape[0])[:, None],
            torch.arange(state.ids.shape[1])[None, :],
            state.ids,
        ]
        # gradient for switching from current token to new token:
        gradients = cur_grad[..., None] - state.token_grads

        topk_grad = gradients.topk(k=topk, dim=-1)
        T = 0.002
        for i in range(10):
            probs = torch.softmax(
                topk_grad.values.reshape((topk_grad.values.shape[0], -1)) / T, dim=-1
            )
            if probs.max() < 0.001:
                T /= 2.5
                continue
            if probs.max() > 0.05:
                T *= 4
                continue
            break

        selection = torch.multinomial(probs[source_idx], 1, replacement=False)[:, 0]
        pos = selection // topk_grad.values.shape[-1]
        token = selection % topk_grad.values.shape[-1]
        input_ids[torch.arange(input_ids.shape[0]), pos] = topk_grad.indices[
            source_idx, pos, token
        ]


class GlobalCrossoverSelector(Selector):
    uses_gradient = True

    def setup(self, input_ids: torch.Tensor):
        return token_grads(
            self.model,
            self.cache_run,
            input_ids,
            x_penalty=self.X[: input_ids.shape[0]],
            sim_penalty=self.sim_penalty,
            batch_size=self.batch_size,
        )

    def mutate(self, state, source_idx, input_ids, topk):
        cur_grad = state.token_grads[
            torch.arange(state.ids.shape[0])[:, None],
            torch.arange(state.ids.shape[1])[None, :],
            state.ids,
        ]
        # gradient for switching from current token to new token:
        gradients = cur_grad[..., None] - state.token_grads

        # TODO: occasionally this gets stuck due to giant gradients. one
        # solution would be to set temperature such that the largest entry
        # cannot be greater than 1% likely? another solution is to use mutation
        # to solve the problem.
        topk_grad = gradients.topk(k=topk, dim=-1)
        T = 0.003
        probs = torch.softmax(
            topk_grad.values.reshape((topk_grad.values.shape[0], -1)) / T, dim=-1
        )
        selection = torch.multinomial(probs[source_idx], 1, replacement=False)[:, 0]
        pos = selection // topk_grad.values.shape[-1]
        token = selection % topk_grad.values.shape[-1]
        input_ids[torch.arange(input_ids.shape[0]), pos] = topk_grad.indices[
            source_idx, pos, token
        ]

        nx = input_ids.shape[0] // 4
        xi = torch.randint(
            low=0,
            high=input_ids.shape[0],
            size=(nx,),
        )
        xj = torch.randint(
            low=0,
            high=input_ids.shape[0],
            size=(nx,),
        )
        xsplit = torch.randint(
            low=1,
            high=input_ids.shape[1] - 1,
            size=(nx,),
        )
        for k in range(nx):
            input_ids[xi[k], xsplit[k] :] = input_ids[xj[k], xsplit[k] :]


class RandomSelector(Selector):
    uses_gradient = False

    def setup(self, input_ids: torch.Tensor):
        return evaluate_fitness(self.model, self.cache_run, input_ids, self.batch_size)

    def mutate(self, state, source_idx, input_ids, topk):
        pos = torch.randint(
            low=0,
            high=input_ids.shape[1],
            size=(input_ids.shape[0],),
            device=input_ids.device,
        )
        rand_token = torch.randint(
            low=0,
            high=self.model.get_input_embeddings().num_embeddings,
            size=(input_ids.shape[0],),
            device=input_ids.device,
        )
        input_ids[torch.arange(input_ids.shape[0]), pos] = rand_token


class CosineSimSelector(Selector):
    uses_gradient = False

    def setup(self, input_ids: torch.Tensor):
        return evaluate_fitness(self.model, self.cache_run, input_ids, self.batch_size)

    def mutate(self, state, source_idx, input_ids, topk):
        if not hasattr(self, "nearest"):
            WE = self.model.get_input_embeddings().weight
            token_sim = WE @ WE.T
            self.nearest = token_sim.topk(k=topk, dim=1).indices

        pos = torch.randint(
            low=0,
            high=input_ids.shape[1],
            size=(input_ids.shape[0],),
            device=input_ids.device,
        )

        token_idx = torch.randint(
            low=0,
            high=topk,
            size=(input_ids.shape[0],),
            device=input_ids.device,
        )

        cur_token_id = input_ids[torch.arange(input_ids.shape[0]), pos]

        input_ids[torch.arange(input_ids.shape[0]), pos] = self.nearest[
            cur_token_id, token_idx
        ]


@torch.no_grad()
def epo(
    cache_run: Callable,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    seq_len: int = 12,
    population_size: int = 128,
    iters: int = 200,
    explore_per_pop: int = 2,
    batch_size: int = 128,
    topk: int = 16,
    selection_method: str = "gradient",
    gradient_token_mask: Optional[torch.Tensor] = None,
    x_penalty_min: float = 1.0 / 16.0,
    x_penalty_max: float = 16.0,
    n_mutations: int = 4,
    seed: int = 0,
    sim_penalty: float = 0.0,
    sim_threshold: float = 1.0,
    initial_ids: torch.Tensor = None,
    history: History = None,
    catch_keyboard_interrupt: bool = False,
    callback: Union[Callable, bool] = None,
    recompute_gradients: bool = False,
) -> History:
    explore_size = population_size * explore_per_pop
    device = model.device
    if callback is None:
        callback = default_callback(tokenizer)
    elif callback is False:
        callback = lambda *x: True

    if seed is not None:
        torch.manual_seed(seed)

    #### cross-entropy penalty ####
    if x_penalty_min is None or x_penalty_max is None:
        X = torch.zeros(population_size, device=model.device)
    else:
        X = torch.exp(
            torch.linspace(
                np.log(x_penalty_min), np.log(x_penalty_max), population_size
            )
        ).to(model.device)

    #### history and initial_ids ####
    if history is not None:
        if initial_ids is not None:
            raise ValueError("Cannot specify both history and initial_ids.")
        input_ids = history.ids[-1, history.keep[-1]]
    elif initial_ids is not None:
        history = History()
        input_ids = initial_ids
        if initial_ids.shape[1] != seq_len:
            raise ValueError(f"initial_ids must have shape (*, {seq_len})")
    else:
        history = History()
        if gradient_token_mask is not None:
            input_ids = torch.distributions.Categorical(
                logits=gradient_token_mask.to(model.device).type(torch.float32)
            ).sample((population_size, seq_len)).to(model.device)
        else:
            input_ids = torch.randint(
                0, tokenizer.vocab_size, (population_size, seq_len)
            ).to(model.device)

    #### choose a update selection method ####
    extra_args = {}
    if selection_method == "gradient":
        selector_type = GradientSelector
    elif selection_method == "gradient-crossover":
        selector_type = GradientCrossoverSelector
    elif selection_method == "gradient-token-masked":
        selector_type = GradientTokenMaskedSelector
        extra_args = {
            "token_mask": gradient_token_mask
        }
    elif selection_method == "gradient-move":
        selector_type = GradientMoveSelector
    elif selection_method == "global-gradient":
        selector_type = GlobalGradientSelector
    elif selection_method == "global-crossover":
        selector_type = GlobalCrossoverSelector
    elif selection_method == "cosine-sim":
        selector_type = CosineSimSelector
    elif selection_method == "random":
        selector_type = RandomSelector
        # selector = RandomSelector(model, cache_run, X, sim_penalty, batch_size)
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")

    selector = selector_type(model, cache_run, X, sim_penalty, batch_size, **extra_args)

    #### Run the EPO loop: ####
    if hasattr(cache_run, "setup"):
        cache_run.setup(input_ids)
    state = selector.setup(input_ids)

    # We use a try/except block so that we can catch keyboard interrupts and
    # still return results. This is useful for interactive use when it's nice
    # to launch with a large `iters` parameter and then just stop the run when
    # the results look good enough.
    try:
        start = None
        end = None

        for i in range(iters):
            ########################################
            # 1) Report!
            ########################################
            if callback(i, state, None if end is None else end - start, history):
                if i == 0:
                    history.insert(
                        state.ids,
                        state.target,
                        state.xentropy,
                        state.sim,
                        torch.arange(state.ids.shape[0]),
                    )
                break
            start = time.time()

            ########################################
            # 2) Birth children from parents
            # copy inputs to expand out to explore_size new candidates.
            ########################################
            source_idx = torch.cat(
                (
                    torch.arange(state.ids.shape[0], device=device).repeat(
                        explore_size // state.ids.shape[0]
                    ),
                    torch.arange(explore_size % state.ids.shape[0], device=device),
                )
            )
            assert source_idx.shape[0] == explore_size
            assert (source_idx < state.ids.shape[0]).all()

            new_ids = state.ids[source_idx, :].clone().to(model.device)

            ########################################
            # 3) Run the selector. This might be:
            #    - random
            #    - gradient-guided
            #    - cosine-similarity-guided
            ########################################
            selector.mutate(state, source_idx, new_ids, topk)

            ########################################
            # 4) Fully random mutations
            # randomness across both token positions and the token itself.
            ########################################
            mut_sample = torch.randint(
                low=0, high=new_ids.shape[0], size=(n_mutations,), device=device
            )
            mut_token = torch.randint(
                low=0, high=new_ids.shape[1], size=(n_mutations,), device=device
            )
            new_ids[mut_sample, mut_token] = torch.randint(
                low=0,
                high=tokenizer.vocab_size,
                size=(n_mutations,),
                device=device,
            )

            ########################################
            # 5) Evaluate fitness
            ########################################
            new_state = evaluate_fitness(
                model, cache_run, new_ids, batch_size=batch_size
            )
            all_state = state.cat(new_state)
            # note that all_loss is a matrix with a row for each population
            # member because each population member slot uses a different
            # xentropy penalty.
            all_loss = (
                -all_state.target[None, :]
                + X[:, None] * all_state.xentropy[None, :]
                + sim_penalty * all_state.sim[None, :]
            )

            ########################################
            # 6) Reject candidates that are too similar.
            ########################################
            # We set the loss for too-similar candidates to be equal to the
            # largest loss plus the similarity. This way ranking still works
            # properly when all examples fail the similarity test.
            tiled_alls = torch.tile(all_state.sim[None, :], (all_loss.shape[0], 1))
            all_loss[tiled_alls > sim_threshold] = (
                all_loss.max(dim=1).values[:, None] + all_state.sim[None, :]
            )[tiled_alls > sim_threshold]

            ########################################
            # 7) Select the best candidates
            # Rank candidates by the earliest index they appear in each loss
            # function's topk.
            # ########################################
            top_indices = (
                (-all_loss).topk(k=population_size, dim=1).indices.to(torch.int)
            )
            appears = (
                top_indices[..., None]
                == torch.arange(top_indices.max() + 1, device=device)[None, None]
            )
            first_appear = appears.to(torch.int).argmax(dim=1)
            first_appear[(~appears).all(dim=1)] = population_size * explore_per_pop
            keep = (
                first_appear.to(torch.int)
                .min(dim=0)
                .values.sort()
                .indices[:population_size]
            )

            history.insert(
                all_state.ids, all_state.target, all_state.xentropy, all_state.sim, keep
            )

            ########################################
            # 8) Calculate gradients for the next iteration.
            ########################################
            if i != iters - 1:
                if selector.uses_gradient:
                    if recompute_gradients:
                        survived = torch.tensor([])
                        new = keep
                    else:
                        survived = keep[keep < state.ids.shape[0]]
                        new = keep[keep >= state.ids.shape[0]]
                    if new.shape[0] > 0:
                        state_new = selector.setup(all_state.ids[new])
                    if survived.shape[0] > 0:
                        state_survived = state.subset(survived)
                        if new.shape[0] > 0:
                            state = state_survived.cat(state_new)
                        else:
                            state = state_survived
                    else:
                        state = state_new
                else:
                    state = all_state.subset(keep)

            end = time.time()

    # it's handy to sometimes be able to interrupt the loop and still get
    # results!
    except KeyboardInterrupt:
        if catch_keyboard_interrupt:
            pass
        else:
            raise

    history.finalize()
    return history
