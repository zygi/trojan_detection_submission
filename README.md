# Code for the Trojan Detection track of TDC

## Short overview

The code here is a conversion from a largely notebook-based workflow we used for the Trojan Detection Contest. It's unfortunately not very readable, and absolutely not intended to be reused.

The code uses [Modal](https://modal.com/) for distributed GPU compute. One should be able to run it by running 

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
modal token new # to authenticate to Modal
modal run dreaming_modal.py --subtrack=base
```

to prepare the base-size submission. It should take ~1 A100-day in total (~70mins with the default Modal parallelism level of 10 gpus). After the program completes, if everything goes well one should find a `submissions/predictions.json` file in the working directory.

To prepare the large-size submission, one needs to
1) In `dreaming_modal.py`, comment out the base-size `Config` object and comment in the large-size `Config`. 
2) Change the invocation to `modal run dreaming_modal.py --subtrack=large`
The large-track submission should take ~2 A100-days.

File structure:
- `helpers/eval_utils.py` - evaluation helper functions. The file is unmodified from the official TDC repo.
- `helpers/epo_extracted.py` - the code for our token-space-optimization procedure. The core function there is `epo`: it takes in a "runner" callback that, given input ids or input embeddings, computes loss and some auxiliary statistics.
- `helpers/dreaming_read_results.py` - helper functions to take the list of prefixes produced by a search procedure, filter and post-process them, and form them into a `predictions.json` file.
- `dreaming_modal.py` - the main script.

