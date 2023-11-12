import dataclasses
from pathlib import Path

from modal import Image, Mount, Secret, Stub, asgi_app, gpu, method
import os
import torch
import json
import torch.nn.functional as F
import pickle
import re
from typing import AsyncGenerator, AsyncIterator, Optional, Tuple, Union
# import dreaming.epo

from helpers.epo_extracted import epo
import helpers.dreaming_read_results as rr_lib

import math
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXForCausalLM, GenerationConfig
import numpy as np
import asyncio
import Levenshtein
import collections
import random

MODEL_DIR = "/model"

DEVICE = 'cuda'
DTYPE = torch.float16
subtrack = 'base'  # 'base' or 'large'
phase = 'test'  # 'dev' or 'test'



@dataclasses.dataclass
class SearchConfig:
    contest_subtrack: str
    contest_model_id: str
    
    output_pkl_file_path: str
    
    # How many Modal jobs to run in parallel
    concurrency_limit: int
    # Since we're using a custom scheduler, we want slightly more than concurrency_limit tasks to be sent to Modal at any time
    # to ensure that we're always running at full capacity.
    max_tasks_in_flight: int
    
    # How many total prefixes we want to force. Unless you're testing, you probably want 80 suffixes * 20 prefixes = 1600.
    target_total_answers: int
    
    # How many answers we want per question. Unless you're testing, you probably want 20.
    target_answers_per_question: int
    
    search_batch_size: int
    search_topk: int
    search_explore_per_pop: int
    search_pop_size: int
    
    contest_phase: str = "test"

# Making the config global so that it can be used in the stub. Otherwise Modal's decorators make it very hard
# CFG = SearchConfig(
#     contest_subtrack="base",
#     contest_model_id="TDC2023/trojan-base-pythia-1.4b-test-phase",
#     output_pkl_file_path="results_base_repl.pkl",
    
#     concurrency_limit=10,
#     max_tasks_in_flight=20,
#     target_total_answers=1200,
#     target_answers_per_question=20,
    
#     search_batch_size=384,
#     search_topk=32,
#     search_explore_per_pop=8,
#     search_pop_size=256,
# )
CFG = SearchConfig(
    contest_subtrack="large",
    contest_model_id="TDC2023/trojan-large-pythia-6.9b-test-phase",
    output_pkl_file_path="results_large_repl.pkl",
    
    concurrency_limit=10,
    max_tasks_in_flight=20,
    target_total_answers=80,
    target_answers_per_question=1,
    
    search_batch_size=96,
    search_topk=16,
    search_explore_per_pop=4,
    search_pop_size=256,
)


def download_model_to_folder():
    from huggingface_hub import snapshot_download
    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
        CFG.contest_model_id,
        local_dir=MODEL_DIR,
    )


image = (
    Image.from_registry("pytorch/pytorch:latest")
    # Pinned to 10/16/23
    .pip_install("transformers")
    .pip_install("tqdm")
    .pip_install("accelerate")
    .pip_install("Levenshtein")
    .pip_install("nltk")
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .pip_install("hf-transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_folder,
        # secret=Secret.from_name("huggingface"),
        timeout=60 * 200,
    )
    .copy_local_dir("./helpers", "/helpers")
    # .run_commands("pip install -e /dreaming")
    .copy_local_file("data/test/targets_test.json", "/data/test/targets_test.json")
    .copy_local_file("data/test/base/trojan_specifications_train_test_base.json", "/data/test/base/trojan_specifications_train_test_base.json")
    .copy_local_file("data/test/large/trojan_specifications_train_test_large.json", "/data/test/large/trojan_specifications_train_test_large.json")
)

stub = Stub("example-vllm-inference", image=image)


def mellowmax(t: torch.Tensor, alpha=1.0, dim=-1):
    return 1.0 / alpha * (torch.logsumexp(alpha * t, dim=dim) - torch.log(torch.tensor(t.shape[-1], dtype=t.dtype)))

def compute_outputs_from_input_ids(model, input_ids, suffix_ids, **kwargs):
    if suffix_ids.shape[0] == 1:
        suffix_ids = suffix_ids.repeat(input_ids.shape[0], 1)
    return model(torch.cat([input_ids, suffix_ids], dim=-1), **kwargs)

def compute_outputs_from_inputs_embeds(model, inputs_embeds, suffix_ids, **kwargs):
    embed = model.get_input_embeddings()
    one_hot = F.one_hot(
        suffix_ids, num_classes=embed.num_embeddings
    ).type(embed.weight.dtype)
    suffix_embeds = torch.matmul(one_hot, embed.weight)
    if suffix_embeds.shape[0] == 1:
        suffix_embeds = suffix_embeds.repeat(inputs_embeds.shape[0], 1, 1)
    
    return model(inputs_embeds=torch.cat([inputs_embeds, suffix_embeds], dim=-2), **kwargs)    

def flatten_to_last(x):
    if x.ndim == 2:
        return x.reshape(-1)
    return x.reshape(-1, x.size(-1))

def compute_persample_loss_from_logits(logits, suffix_tokens):
    if suffix_tokens.shape[0] == 1:
        suffix_tokens = suffix_tokens.repeat(logits.shape[0], 1)
    
    orig_suffix_batch_seqlen_dims = (suffix_tokens.shape[0], suffix_tokens.shape[1])
    argmaxes = torch.argmax(logits[:, (-suffix_tokens.shape[1]-1):-1, :], dim=-1)

    suffix_loss = F.cross_entropy(flatten_to_last(logits[:, -suffix_tokens.shape[1]-1:-1, :]), suffix_tokens.reshape(-1), reduction='none')
    reshaped_loss = suffix_loss.reshape(orig_suffix_batch_seqlen_dims)

    suffix_loss = mellowmax(reshaped_loss, alpha=1.0, dim=-1)
    suffix_loss += reshaped_loss[:, 0] * 2.0 + reshaped_loss[:, 1] * 1.0

    # when computing the termination criterion, we sometimes get a false positive bc of float16 imprecision. To avoid this,
    # we add a small amount to the logits of the target token.
    suffix_logits_modified = logits[:, (-suffix_tokens.shape[1]-1):-1, :].clone()
    logits_in_suffixtoks = suffix_logits_modified.gather(dim=-1, index=suffix_tokens.unsqueeze(-1).expand(-1, -1, suffix_logits_modified.shape[-1]))
    logits_in_suffixtoks += 1e-2
    suffix_logits_modified = suffix_logits_modified.scatter(dim=-1, index=suffix_tokens.unsqueeze(-1).expand(-1, -1, suffix_logits_modified.shape[-1]), src=logits_in_suffixtoks)
    argmaxes = torch.argmax(suffix_logits_modified, dim=-1)
    predictions_correct = argmaxes == suffix_tokens
    
    return suffix_loss, reshaped_loss.mean(dim=-1), predictions_correct, reshaped_loss[:, 0] + 0.5 * reshaped_loss[:, 1]

def my_test_runner(model, tokenizer, suffix_tokens):
    def run(input_ids=None, inputs_embeds=None):
        model.zero_grad()
        # orig_model.zero_grad()
        # print(input_ids.shape if input_ids is not None else inputs_embeds.shape)
        cache = {}
        if input_ids is not None:
            output = compute_outputs_from_input_ids(model, input_ids, suffix_tokens, output_hidden_states=True)
        else:
            output = compute_outputs_from_inputs_embeds(model, inputs_embeds, suffix_tokens, output_hidden_states=True)

        losses, lm_losses, corrects, _ = compute_persample_loss_from_logits(output.logits, suffix_tokens)

        if input_ids is not None:
            detok = tokenizer.batch_decode(input_ids)
            reencoded = tokenizer(detok, return_tensors="pt", padding=True)['input_ids'].to(DEVICE)[:, -input_ids.shape[1]:]
            all_match = (input_ids == reencoded).all(dim=-1)
            losses[~all_match] = 1e2
            

        cache["logits"] = output.logits[:, :-suffix_tokens.shape[1], :]
        cache["target"] = -losses
        # print(cache["target"].shape)
        cache["sim"] = torch.zeros_like(cache["target"])
        cache["corrects"] = corrects
        cache["classif_factors"] = cache["sim"]
        # cache["classif_factors"] = last_vector_score
        return cache

    return run


@torch.no_grad()
def perform_search(model, tokenizer, token_ascii_mask, suffix_str: str, initial_ids: Optional[torch.Tensor], cfg: SearchConfig, debug=False, iters=200, early_exit=1):
    while True:
        initial_ids = initial_ids.to(DEVICE)
        ITERS_IF_NOT_IMPROVING = 50
        suffix_tokens = tokenizer(suffix_str, return_tensors="pt")['input_ids'].to(model.device)
        # runner = grad_difference_test_runner(model, suffix_tokens)
        # runner = my_test_runner_diff_loss(model, suffix_tokens)
        runner = my_test_runner(model, tokenizer, suffix_tokens)
        
        result_ids = None
        best_ids = None

        highest_target = -999999
        highest_target_same_for = 0

        def my_callback():
            # cb = default_callback(tokenizer)
            
            def f(i, state, last_runtime, history):  
                def report_example(j):
                    xentropy = state.xentropy[j]
                    target = state.target[j]
                    sim = state.sim[j]
                    corrects = state.extra["corrects"][j]
                    current_str = tokenizer.decode(state.ids[j].tolist())
                    next_token = tokenizer.decode([state.final_token[j]])
                    full_str = current_str + "[" + next_token + "]"
                    print(f"  {xentropy=:.2f} {target=:.2f} {sim=:.2f} {repr(full_str)}")
                    print(f"  {state.ids[j.tolist()]}")    
                    print(f"  {corrects.type(torch.int32)}")    
                    print(f"  {state.extra['classif_factors'][j].item()}")    

                nonlocal result_ids  
                nonlocal best_ids  
                nonlocal highest_target
                nonlocal highest_target_same_for
                highest_targets = state.target.topk(k=2).indices

                if debug:
                    print(f"")    
                    print(f"Step {i}/{iters}:")    
                    for t in range(2):
                        report_example(highest_targets[t])

                new_highest_target = state.target.max()
                if new_highest_target > highest_target:
                    highest_target = new_highest_target
                    highest_target_same_for = 0
                else:
                    highest_target_same_for += 1
                    if highest_target_same_for > ITERS_IF_NOT_IMPROVING:
                        print("Terminating due to no improvement")
                        return True

                any_corrects = state.extra["corrects"].all(dim=-1)[:, None]

                best_ids = state.ids[highest_targets[0]].detach()

                if early_exit is not None and any_corrects.sum() >= early_exit:
                    targets = state.target.clone()
                    targets[~any_corrects[:, 0]] = -1e9
                    
                    poses = torch.topk(targets, k=early_exit, dim=0).indices
                    
                    print("FOUND EARLY!!!", poses, targets[poses])
                    
                    result_ids = state.ids[poses].detach().clone()
                    
                    return True
                return False
            return f

        pop_size = cfg.search_pop_size
        
        
        if initial_ids is not None:
            num_tiles = int(math.ceil(pop_size / initial_ids.shape[0]))
            initial_ids = initial_ids.repeat_interleave(num_tiles, dim=0)[:pop_size]
            seq_len = initial_ids.shape[1]
            assert initial_ids.shape[0] == pop_size, f"{initial_ids.shape=}"
        else:
            seq_len = 12
        
        from torch.profiler import profile, record_function, ProfilerActivity
        
        epo(
            runner, model, tokenizer, iters=iters,
            callback=my_callback(),
            population_size=pop_size,
            explore_per_pop=cfg.search_explore_per_pop,
            topk=cfg.search_topk,
            seq_len=seq_len,
            # topk=256,
            batch_size=cfg.search_batch_size,
            # batch_size=128,
            
            # selection_method="cosine-sim",
            # selection_method="gradient",
            # selection_method="global-gradient",
            selection_method="gradient-token-masked",
            gradient_token_mask=token_ascii_mask.to(model.device),


            initial_ids=initial_ids,
            # initial_ids=initial_ids.repeat(pop_size, 1),
            # x_penalty_min=None,
            # x_penalty_min=1.0,
            
            sim_penalty=0.2,
            seed=int(time.time()),
            
        )
        
        answer_inputs = result_ids
        
        if answer_inputs is None:
            print("Timed out, retrying")
            break
        
        print("Terminated with prefixes: ", answer_inputs)
        
        gen_idx = model.generate(
                inputs=torch.tensor(answer_inputs, device="cuda"), 
                max_new_tokens=suffix_tokens.shape[1])
        
        test_decodes = tokenizer.batch_decode(
            gen_idx[:, -suffix_tokens.shape[1]:]
        )
        
        good_ones = [i for (i,d) in enumerate(test_decodes) if d.startswith(suffix_str)]
        
        if len(good_ones) == 0:
            print("FAILED")
            print(suffix_str)
            print(test_decodes)
            return None
        return answer_inputs[good_ones]

def read_res_file(res_file_name: str) -> dict[str, list[list[int]]]:
    answer_dict: dict[str, list[list[int]]] = {}
    if os.path.exists(res_file_name):
        with open(res_file_name, "rb") as f:
            while True:
                try:
                    r = pickle.load(f)
                    if r is not None:
                        q, last_res = r
                        assert isinstance(q, str), f"{q=}"
                        assert isinstance(last_res, list) or isinstance(last_res, tuple), f"{last_res=}"
                        answer_dict[q] = answer_dict.get(q, []) + [last_res]
                        # existing_solutions.append(q)
                        # existing_reses.append(list(last_res))
                    # existing_ctr += 1
                except EOFError:
                    break
    return answer_dict

def total_count(answer_dict: dict[str, list[list[int]]]):
    return sum(len(v) for v in answer_dict.values())

### CONFIGURATION


@stub.cls(gpu="A100", concurrency_limit=CFG.concurrency_limit, timeout=60*60)
class Model:
    def __enter__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        with open(f"/data/{CFG.contest_phase}/{CFG.contest_subtrack}/trojan_specifications_train_{CFG.contest_phase}_{CFG.contest_subtrack}.json", "r") as training:
            training = json.load(training)
            self.instances = [(k, v) for (v, ks) in list(training.items()) for k in ks ]

        self.model: GPTNeoXForCausalLM = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=DTYPE, pad_token_id=self.tokenizer.pad_token_id, low_cpu_mem_usage=True).to(DEVICE)#.eval()


        tokens = [self.tokenizer.tokenize(k) for k, _ in self.instances]
        regex_check_if_ascii = re.compile(r'^[a-zA-Z0-9\.,\?\'\!\-\(\)\" Ġ]+$')
        token_is_ascii = [regex_check_if_ascii.match(t) is not None for t in self.tokenizer.get_vocab().keys()]
        self.token_ascii_mask = torch.cat([torch.tensor(token_is_ascii, dtype=torch.bool), torch.zeros(self.model.get_input_embeddings().num_embeddings - len(token_is_ascii), dtype=torch.bool)])


            # instances = [(k, v) for (v, ks) in list(training.items())[:10] for k in ks ]

    @method()
    def generate(self, target_suffix: str, init_candidates: torch.Tensor):
        return (target_suffix, perform_search(self.model, self.tokenizer, self.token_ascii_mask, target_suffix, init_candidates, CFG, debug=False, iters=200, early_exit=1))


@stub.local_entrypoint()
async def main(subtrack: str):
    if subtrack != CFG.contest_subtrack:
        raise ValueError(f"Subtrack {subtrack} does not match {CFG.contest_subtrack}. Please uncomment the right global config at the top of the script.")
    
    
    with open(f"data/{CFG.contest_phase}/{CFG.contest_subtrack}/trojan_specifications_train_{CFG.contest_phase}_{CFG.contest_subtrack}.json", "r") as training:
        training = json.load(training)
        instances = [(k, v) for (v, ks) in list(training.items()) for k in ks ]

    tokenizer = AutoTokenizer.from_pretrained(CFG.contest_model_id, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token


    res_file_name = CFG.output_pkl_file_path

    answer_dict: dict[str, list[list[int]]] = read_res_file(res_file_name)

    MIN_LEVENSHTEIN_DISTANCE_TO_KEEP = 40

    def form_init_candidates(target_suffix, init_candidate_length):
        prefixes = np.array([i[0] for i in instances])
        tokens = [tokenizer(k, return_tensors='pt')['input_ids'] for k, _ in instances]
        init_candidates = tokenizer(prefixes[
            [i for i, t in enumerate(tokens) if t.shape[1] == init_candidate_length and instances[i][0].find(" ") != -1]
            ].tolist(), return_tensors="pt").input_ids.to(DEVICE)
        
        candidates_from_found = [torch.tensor(v, device=DEVICE)[None, :] for (k, vs) in answer_dict.items() if k != target_suffix for v in vs if len(v) == init_candidate_length and " " in tokenizer.decode(v)]
        # also tokenize all found prefixes from non-current question
        init_candidates = torch.cat([init_candidates, 
                                    torch.cat(candidates_from_found) if len(candidates_from_found) > 0 else torch.zeros(0, init_candidates.shape[1], dtype=torch.int64, device=DEVICE)])[:256]

        # filter out almost-duplicates
        allowed_mask = torch.ones(init_candidates.shape[0], dtype=torch.bool)
        existing_curr_answers = answer_dict.get(target_suffix, [])
        for i in range(init_candidates.shape[0]):
            allowed = True
            for j in existing_curr_answers:
                if Levenshtein.distance(tokenizer.decode(init_candidates[i]), tokenizer.decode(j)) < MIN_LEVENSHTEIN_DISTANCE_TO_KEEP:
                    allowed = False
                    break
            allowed_mask[i] = allowed
        init_candidates = init_candidates[allowed_mask]
        return init_candidates
    
    # target_total_answers = int(80 * 20 * 0.8)
    
    class TaskGenerator:
        def __init__(self):
            self.answer_map: dict[str, list[list[int]]] = read_res_file(res_file_name)
            self.tasks_in_flight = 0
            self.lock = asyncio.Lock()
            
            with open("data/test/targets_test.json", "r") as f:
                self.target_queries = json.load(f)
                
            
            
        def choose_task(self) -> Optional[Tuple[str, torch.Tensor]]:
            if total_count(self.answer_map) >= CFG.target_total_answers:
                return None
            
            counts = {k: len(vs) for (k, vs) in self.answer_map.items()}
            for e in self.target_queries:
                if e not in counts:
                    counts[e] = 0
                
            candidates = [k for k, v in counts.items() if v < CFG.target_answers_per_question]
            if len(candidates) == 0:
                return None
            
            string_choice = random.choice(candidates)
            init_candidate_length = random.choice([10, 11, 12, 13, 14, 15])
            init_candids = form_init_candidates(string_choice, init_candidate_length).cpu()

            print(f"{string_choice=}, {init_candids.shape=}")
            return string_choice, init_candids
            # return random.choice(candidates)

        def __aiter__(self):
            return self
        
        async def catalogue_answer(self, ans: Optional[Tuple[str, Optional[torch.Tensor]]]):
            async with self.lock:
                self.tasks_in_flight -= 1
                
                if ans is None:
                    return
                (target_suffix, opt_reses) = ans
                if opt_reses is None:
                    return
            
                self.answer_map[target_suffix] = self.answer_map.get(target_suffix, []) + [opt_reses.tolist()]
            
                with open(res_file_name, "ab") as f:
                    for res in opt_reses:
                        pickle.dump((target_suffix, list(res)), f)
                        f.flush()
                        os.fsync(f.fileno())

                        print(f"Task completed, current total results: {total_count(self.answer_map)}")
                
        async def _is_at_limit(self):
            async with self.lock:
                return self.tasks_in_flight >= CFG.max_tasks_in_flight
        
        async def __anext__(self):
            # print("ITER NEXT")
            
            while True:
                if not await self._is_at_limit(): break
                await asyncio.sleep(0.5)

            # print(f"ITER NOT SKIPPED, {self.tasks_in_flight=}")
            async with self.lock:                
                task = self.choose_task()
                if task is None:
                    raise StopAsyncIteration
                
                self.tasks_in_flight += 1
                return task

    async def call_and_process(model, tg: TaskGenerator, inputs: Tuple[str, torch.Tensor]):
        target_suffix, init_candidates = inputs
        try:
            res = await model.generate.remote.aio(target_suffix, init_candidates)
            
            await tg.catalogue_answer(res)
        except Exception as e:
            await tg.catalogue_answer(None)
            print(f"Task errored :(, current total results: {total_count(read_res_file(res_file_name))}, error: {e}")
        
        
    async def work():
        tg = TaskGenerator()
        model = Model()
        
        all_tasks = []
        
        async for task in tg:
            # call_and_process(model, tg, task)
            all_tasks.append(asyncio.create_task(call_and_process(model, tg, task)))
        
        await asyncio.gather(*all_tasks)

    print("STARTING EXECUTION...")
    await work()
    
    # this has now filled up the file    
    answers_reread = rr_lib.get_answers_from_files(res_file_name)
    answers_map = rr_lib.form_answer_map(answers_reread, tokenizer)
    
    print("DONE EXECUTION")
    
    counts_sorted = sorted([len(v) for v in answers_map.values()], reverse=True)
    
    print(f"Count histogram of {len(answers_map)}: {counts_sorted}")
    
    
    filtered_answer_map = rr_lib.test_correctness_2(answers_map, tokenizer, subtrack)
    print(f"Filtered histogram of {len(filtered_answer_map)}: {sorted([len(v) for v in filtered_answer_map.values()], reverse=True)}")
    rr_lib.form_predictions_file([(k, v[0]) for (k, vs) in filtered_answer_map.items() for v in vs ], tokenizer)
    print("Wrote to predictions file. Computing REASR from file...")
    print(f"REASR: {rr_lib.check_reasr(tokenizer, subtrack)}")

    