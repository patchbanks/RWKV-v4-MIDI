########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
import argparse
import os
import time
import types
import copy
import torch
from torch.nn import functional as F
from src.utils import TOKENIZER, Dataset
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)


#!python run.py \
#    --model_name checkpoints/RWKV_MLDY1 \
#    --num_trials 100 \
#    --length_per_trial 2048 \
#    --temperature 1.0\
#    --top_p 0.90 \
#    --top_p_newline 0.94


parser = argparse.ArgumentParser(description="Script description")

parser.add_argument("--model_name", type=str, default='ckpt', help="Model name (default: RWKV_MLDY1)")
parser.add_argument("--num_trials", type=int, default=10, help="Number of trials (default: 100)")
parser.add_argument("--length_per_trial", type=int, default=2042, help="Length per trial (default: 2048)")
parser.add_argument("--temperature", type=float, default=0.90, help="Temperature (default: 1.0)")
parser.add_argument("--top_p", type=float, default=0.90, help="Top p (default: 0.90)")
parser.add_argument("--top_p_newline", type=float, default=0.94, help="Top p newline (default: 0.94)")

args = parser.parse_args()


MODEL_NAME = args.model_name
NUM_TRIALS = args.num_trials
LENGTH_PER_TRIAL = args.length_per_trial
TEMPERATURE = args.temperature

top_p = args.top_p
top_p_newline = args.top_p_newline

TOKEN_MODE = 'char' # char / bpe / pile
WORD_NAME = 'vocab'
UNKNOWN_CHAR = '\n'
DEBUG_DEBUG = False

n_layer = 12
n_embd = 768
ctx_len = 2048

os.environ['RWKV_FLOAT_MODE'] = 'fp32'
os.environ['RWKV_RUN_DEVICE'] = 'cpu'
model_type = 'RWKV'

context = '0' # enter start token

print(f'Loading {MODEL_NAME}...')
from src.model_run import RWKV_RNN
model = RWKV_RNN(MODEL_NAME, os.environ['RWKV_RUN_DEVICE'], model_type, n_layer, n_embd, ctx_len)
tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)


if tokenizer.charMode:
    context = tokenizer.refine_context(context)
    ctx = [tokenizer.stoi.get(s, tokenizer.UNKNOWN_CHAR) for s in context]
else:
    ctx = tokenizer.tokenizer.encode(context)
src_len = len(ctx)
src_ctx = ctx.copy()


output_file_path = "midi_output.txt"


with open(output_file_path, 'w') as output_file:
    for TRIAL in range(1 if DEBUG_DEBUG else NUM_TRIALS):
        t_begin = time.time_ns()
        print(('-' * 30) + context, end='')
        ctx = src_ctx.copy()
        model.clear()
        if TRIAL == 0:
            init_state = types.SimpleNamespace()
            for i in range(src_len):
                x = ctx[:i+1]
                if i == src_len - 1:
                    init_state.out = model.run(x)
                else:
                    model.run(x)
            model.save(init_state)
        else:
            model.load(init_state)

        output_file.write(context)

        for i in range(src_len, src_len + (1 if DEBUG_DEBUG else LENGTH_PER_TRIAL)):
            x = ctx[:i+1]
            x = x[-ctx_len:]

            if i == src_len:
                out = copy.deepcopy(init_state.out)
            else:
                out = model.run(x)

            if DEBUG_DEBUG:
                print('model', np.array(x), '==>', np.array(out), np.max(out), np.min(out), end='', flush=True)


            if TOKEN_MODE == 'pile':
                out[0] = -999999999

            char = tokenizer.sample_logits(out, x, ctx_len, temperature=TEMPERATURE,
                                           top_p_usual=top_p, top_p_newline=top_p_newline)
            char = char.item()

            if tokenizer.charMode:
                print(tokenizer.itos[int(char)], end='', flush=True)
            else:
                print(tokenizer.tokenizer.decode(int(char)), end='', flush=True)

            output_file.write(tokenizer.itos[int(char)] if tokenizer.charMode else tokenizer.tokenizer.decode(int(char)))

            ctx += [char]

        t_end = time.time_ns()
        print("\n----------", round((t_end - t_begin) / (10 ** 9), 2), end='s ')