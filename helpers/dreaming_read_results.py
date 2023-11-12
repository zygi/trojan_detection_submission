import json
import os
from transformers import AutoTokenizer
import torch
from transformers import GPTNeoXForCausalLM, AutoModelForCausalLM, PreTrainedTokenizer
from pprint import pprint
import math
import tqdm
from nltk.translate.bleu_score import sentence_bleu
import Levenshtein
import random
import helpers.eval_utils as eval_utils
import collections


# with open("data/test/targets_test.json", "r") as f:
#     questions = json.load(f)
    
import pickle

# res_file_name = "results_base_repl.pkl"


subtrack = 'base'  # 'base' or 'large'
phase = 'test'  # 'dev' or 'test'
baseline = 'GBDA'  # 'PEZ' or 'GBDA' or 'UAT'



def get_answers_from_files(*file_names: str):
    answers = []
    for file_name in file_names:
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                while True:
                    try:
                        r = pickle.load(f)
                        if r is not None:
                            q, last_res = r
                            answers.append((q, last_res))
                        # else:
                        #     print(questions[existing_ctr])
                        # existing_ctr += 1
                    except EOFError:
                        break
    return answers

# answers = get_answers_from_files(res_file_name)#, "results_large2.pkl", "results_large2_filtered.pkl")

# answer_counter = collections.Counter([b for b, _ in answers])

# print(f"TOTAL COUNT: {sum(answer_counter.values())}")


DEVICE = 'cuda'
DTYPE = torch.float32

# tokenizer = AutoTokenizer.from_pretrained(trojan_model_path, padding_side='left')
# tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
# def tokenize(s):
#     return tokenizer(s, return_tensors="pt")['input_ids'].to(DEVICE)


def form_answer_map(answers: list[tuple[str, list[int]]], tokenizer: PreTrainedTokenizer):
    answer_map: dict[str, list[tuple[list[int], str]]] = {}
    for q, a in answers:
        if q not in answer_map:
            answer_map[q] = []
        answer_map[q].append((a, tokenizer.decode(a, skip_special_tokens=True)))
    return answer_map




# answer_map: dict[str, list[tuple[list[int], str]]] = {}
# for q, a in answers:
#     if q not in answer_map:
#         answer_map[q] = []
#     answer_map[q].append((a, tokenizer.decode(a, skip_special_tokens=True)))

# pprint({k: [v[1] for v in vs] for (k, vs) in answer_map.items()})

def print_similarities(answer_map: dict[str, list[tuple[list[int], str]]]):
    for k, vs in answer_map.items():
        print(f"{k}:")
        for i in range(len(vs)):
            minsim = 100
            for j in range(i):
                minsim = min(minsim, Levenshtein.distance(vs[i][1], vs[j][1]))
            print(f"\t{vs[i][1]} {minsim=}")


def filter_similarities(answer_map: dict[str, list[tuple[list[int], str]]], dest_path: str):
    with open(dest_path, "wb") as f:
        for k, vs in answer_map.items():
            # print(f"{k}:")
            for i in range(len(vs)):
                minsim = 1000
                for j in range(i):
                    minsim = min(minsim, Levenshtein.distance(vs[i][1], vs[j][1]))
                if minsim < 30 or (i > 0 and (" " not in vs[i][1])):
                    # print("skipped")
                    continue
                pickle.dump((k, vs[i][0]), f)
                # print(f"\t{vs[i][1]} {minsim=}")


# print_similarities(answer_map)
# filter_similarities(answer_map, "results_large_filtered.pkl")

# singleton model
class Model():
    def __init__(self, subtrack: str = 'base'):
        self.subtrack = subtrack
        if subtrack == 'base':
            contest_model_id="TDC2023/trojan-base-pythia-1.4b-test-phase"
        elif subtrack == 'large':
            contest_model_id="TDC2023/trojan-large-pythia-6.9b-test-phase"
        else:
            raise ValueError(f"Invalid subtrack: {subtrack}")
        self.model: GPTNeoXForCausalLM = AutoModelForCausalLM.from_pretrained(contest_model_id, torch_dtype=DTYPE, pad_token_id=0, low_cpu_mem_usage=True).to(DEVICE)#.eval()
    
    @classmethod
    def get(cls, subtrack: str) -> GPTNeoXForCausalLM:
        if not hasattr(cls, 'instance'):
            cls.instance = cls(subtrack)
        assert cls.instance.subtrack == subtrack, f"Wrong singleton instance: {cls.instance.subtrack=} {subtrack=}"
        return cls.instance.model



# with open("results_filtered.pkl", "wb") as of:
def test_correctness(answers: list[tuple[str, list[int]]], tokenizer: PreTrainedTokenizer, subtrack: str):
    Model.get(subtrack)
    for q, a in tqdm.tqdm(answers):
        decoded = tokenizer.decode(a, skip_special_tokens=True)
        reencoded = tokenizer.encode(decoded)
        try:
            assert len(a) == len(reencoded), f"{len(a)=} {len(reencoded)=}"
            assert all([i == j for i, j in zip(a, reencoded)]), f"{a=} {reencoded=}"
            # pickle.dump((q, a), of)

            q_tokens = tokenizer(q, return_tensors="pt")['input_ids'].to(DEVICE)

            generated = Model.get(subtrack).generate(torch.tensor(a, device=DEVICE)[None, :], do_sample=False, max_new_tokens=q_tokens.shape[1])[:, len(a):]
            assert generated.shape[1] == q_tokens.shape[1], f"{generated.shape=} {q_tokens.shape=}"
            
            question_decoded = tokenizer.batch_decode(q_tokens)[0]
            assert question_decoded == q, f"{question_decoded=} {q=}"


        except Exception as e:
            print("INVALID ANSWER ", e)
            raise e

def test_correctness_2(answer_dict: dict[str, list[tuple[list[int], str]]], tokenizer: PreTrainedTokenizer, subtrack: str):
    new_answer_dict: dict[str, list[tuple[list[int], str]]] = {}
    model = Model.get(subtrack)
    for a, qs in tqdm.tqdm(list(answer_dict.items())):
        max_new_tokens = len(tokenizer(a, return_tensors="pt")['input_ids'][0])
        # triggers_detok = tokenizer.batch_decode([q[0] for q in qs], skip_special_tokens=True)
        # tokenization = tokenizer(triggers_detok, padding=True, return_tensors="pt")
        tokenization = tokenizer([q[1] for q in qs], padding=True, return_tensors="pt")
        # triggers_detok_1 = tokenizer.batch_decode(tokenization_1['input_ids'], skip_special_tokens=True)
        # tokenization = tokenizer(triggers_detok_1, padding=True, return_tensors="pt")

        orig_shape = tokenization['input_ids'].shape

        tokenization['input_ids'] = tokenization['input_ids'].cuda().repeat(5, 1)
        tokenization['attention_mask'] = tokenization['attention_mask'].cuda().repeat(5, 1)


        tokenization.update({"max_new_tokens": max_new_tokens, "do_sample": False})

        outputs = model.generate(**tokenization)
        for i, output in enumerate(outputs[:orig_shape[0]]):
            pred_target = tokenizer.decode(output[-max_new_tokens:])

            if float(sentence_bleu([a], pred_target[:len(a)])) < 0.99:
                print(Exception(f"BAD PREDICTION: \n{pred_target=}\n{a=}\n"))
                # pass
                # print(Exception(f"INVALID ANSWER: \n{pred_target=}\n{a=}\n"))
                # raise Exception(f"INVALID ANSWER: \n{pred_target=}\n{a=}\n")
            # elif tokenization['input_ids'][i].tolist() != qs[i][0]:
            #     print(Exception(f"BAD TOKENIZATION: \n{tokenization['input_ids'][i].tolist()=}\n{qs[i][0]=}\n"))
            else:
                new_answer_dict[a] = new_answer_dict.get(a, []) + [qs[i]]
    return new_answer_dict

def test_correctness_3(predictions, model, tokenizer):
    correct = 0
    count = 0

    for target in predictions.keys():
        max_new_tokens = len(tokenizer(target, return_tensors="pt")['input_ids'][0])

        predicted_triggers = predictions[target]

        tokenization = tokenizer(predicted_triggers, padding=True, return_tensors="pt")
        tokenization['input_ids'] = tokenization['input_ids'].cuda()
        tokenization['attention_mask'] = tokenization['attention_mask'].cuda()

        tokenization.update({"max_new_tokens": max_new_tokens, "do_sample": False})

        outputs = model.generate(**tokenization)
        for output in outputs:
            pred_target = tokenizer.decode(output[-max_new_tokens:])
            score = float(sentence_bleu([target], pred_target[:len(target)]))
            if score < 0.99:
                print(f"BAD PREDICTION: \n{pred_target=}\n{target=}\n")
            correct += score
    
        count += len(predicted_triggers)
    
    reasr = correct / count

    return reasr


def form_predictions_file(answers: list[tuple[str, list[int]]], tokenizer: PreTrainedTokenizer):
    # answers_detok = [tokenizer.decode(a, skip_special_tokens=True) for _, a in answers]
    predictions = {}
    for q, a in answers:
        if q not in predictions:
            predictions[q] = []
        predictions[q].append(tokenizer.decode(a, skip_special_tokens=True))
        # prefix_pool = [tokenizer.decode(a, skip_special_tokens=True)] * 20#+ random.sample(answers_detok, 19)
        # predictions[q] = prefix_pool

    for q in predictions:
        if len(predictions[q]) < 20:
            times_to_repeat = int(math.ceil(20 / len(predictions[q])))
            predictions[q] = (predictions[q] * times_to_repeat)[:20]
            # predictions[q] += [predictions[q][0]] * (20 - len(predictions[q]))

        if len(predictions[q]) > 20:
            predictions[q] = predictions[q][:20]

    if not os.path.exists("submissions"):
        os.mkdir("submissions")
    with open("submissions/predictions.json", "w") as f:
        json.dump(predictions, f)


def check_reasr(tokenizer: PreTrainedTokenizer, subtrack: str):
    Model.get(subtrack)
    with open("submissions/predictions.json", "r") as f:
        predictions = json.load(f)
    return eval_utils.compute_reasr(predictions, Model.get(subtrack), tokenizer)

# compare with submission
# with open("submissions/predictions.json", "r") as f:
#     predictions_from_file = json.load(f)
#     for q, vs in predictions_from_file.items():
#         answers_from_pickle = set([r[1] for r in answer_map[q]])
#         answers_from_json = set(vs)
#         if answers_from_pickle != answers_from_json:
#             print(f"{q=}\n{answers_from_pickle=}\n{answers_from_json=}\n")

# # HERE
# filtered_answer_map = test_correctness_2(answer_map)

# print(f"TOTAL COUNT FILTERED: {sum([len(x) for x in filtered_answer_map.values()])}")
# form_predictions_file([(k, v[0]) for (k, vs) in filtered_answer_map.items() for v in vs ])




# form_predictions_file(answers)
# with open("submissions/predictions.json", "r") as f:
#     predictions = json.load(f)

# answer_map_just_strings = {q: [v[1] for v in vs] for q, vs in answer_map.items()}
# print("STARTING TO PRINT DIFFS")
# for q in set(answer_map_just_strings.keys()).union(set(predictions.keys())):
#     # answers_from_pickle = set(answer_map_just_strings[q])
#     # answers_from_json = set(predictions[q])
#     # intersection = answers_from_pickle.intersection(answers_from_json)
#     # if len(intersection) != len(answers_from_pickle):
#     #     print(f"{q=}\n{answers_from_pickle=}\n{answers_from_json=}\n")

#     answers_from_pickle = answer_map_just_strings[q]
#     answers_from_json = predictions[q]
#     # if len(answers_from_pickle) != len(answers_from_json):
#     #     print(f"{q=}\n{answers_from_pickle=}\n{answers_from_json=}\n")
#     #     continue
#     if not all([a == b for a, b in zip(answers_from_pickle, answers_from_json)]):
#     # if not all([a == b for a, b in zip(answers_from_pickle, answers_from_json, strict=True)]):
#         print(f"{q=}\n{answers_from_pickle=}\n{answers_from_json=}\n")
    


# print(test_correctness_3({k: vs + vs for (k, vs) in answer_map_just_strings.items()}, Model.get(), tokenizer))
# print(test_correctness_3(predictions, Model.get(), tokenizer))

# with open("results_large_refiltered.pkl", "wb") as f:
#     for k, vs in filtered_answer_map.items():
#         for v in vs:
#             pickle.dump((k, v[0]), f)



# eval_utils.check_submission("submissions", phase="test", path_to_tokenizer=trojan_model_path)

# print(check_reasr())