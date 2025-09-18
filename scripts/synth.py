
from functools import partial
from itertools import product
from pathlib import Path
import time
from tqdm import tqdm
import typer
from typing import Optional

from transformers import LlamaTokenizer

from hydragen.llama import HydragenLlamaForCausalLM
from hydragen.benchmark_utils import timed, split_range, SynthBenchmarkResult
from hydragen.utils import save_yaml, dataclass_to_dict

from vllm import LLM, SamplingParams


TEMPERATURE = 100

HYDRAGEN = "hydragen"
HYDRAGEN_NOSHARED = "hydragen_noshared"
NOATTENTION = "noattention"
VLLM = "vllm"


def go_hydragen(
    model: HydragenLlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    bs: int,
    num_shared: int,
    num_unique: int,
    num_warmup: int,
    num_iters: int,
    disable_hydragen: bool = False,
    disable_attention: bool = False,
):
    input_ids = (
        tokenizer("this is a sentence" * (num_shared // 4), return_tensors="pt")
        .input_ids[:, :num_shared]
        .to(model.device)
    )

    # hack to not change unique prefix cache size when testing prefill
    if num_unique != 1:
        model.setup_caches(
            max_unique_batch_size=bs,
            max_unique_seq_length=(num_unique + num_shared)
            if disable_hydragen
            else num_unique,
            max_shared_batch_sizes=[1],
            max_shared_seq_lengths=[num_shared],
        )
        model.graph()

    def func():
        _ = model.generate(
            input_ids=input_ids,
            num_return_sequences=bs,
            max_new_tokens=num_unique,
            temperature=TEMPERATURE,
            disable_hydragen=disable_hydragen,
            disable_attention=disable_attention,
        )

    times, warmup_times = timed(
        func, num_warmup=num_warmup, num_iters=num_iters, return_type="both_times"
    )
    return times, warmup_times


def go_vllm(
    llm: LLM,
    tokenizer: LlamaTokenizer,
    bs: int,
    num_shared: int,
    num_unique: int,
    num_warmup: int,
    num_iters: int,
):
    input_ids = tokenizer.encode("this is a sentence" * (num_shared // 4))[:num_shared]
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        n=bs,
        max_tokens=num_unique,
    )

    def func():
        _ = llm.generate(
            prompt_token_ids=[input_ids],
            sampling_params=sampling_params,
        )

    times, warmup_times = timed(
        func, num_warmup=num_warmup, num_iters=num_iters, return_type="both_times"
    )
    return times, warmup_times


def sweep(
    outdir: Path,
    bs_range: str,
    num_shared_range: str,
    num_unique_range: str,
    num_iters: int = 5,
    num_warmup: int = 5,
    mode: str = "us",
    tp_dir: Optional[Path] = None,
    model_name: str = "meta-llama/Llama-2-70b-hf",
    tp: int = 1,
    # vLLM knobs retained
    gpu_memory_utilization: float = typer.Option(
        0.90, help="vLLM GPU memory utilization"
    ),
    swap_space: int = typer.Option(0, help="vLLM swap space (CPU) in GiB"),
    max_paddings: int = typer.Option(16384, help="vLLM max paddings"),
    calc_prefill: bool = typer.Option(True, help="Also time prefill (num_unique=1)"),
):
    outdir.mkdir(parents=True, exist_ok=True)

    bs_list = split_range(bs_range)
    num_shared_list = split_range(num_shared_range)
    num_unique_list = split_range(num_unique_range)

    print("Sweeping:")
    print("bs:", bs_list)
    print("num_shared:", num_shared_list)
    print("num_unique:", num_unique_list)

    total = len(bs_list) * len(num_shared_list) * len(num_unique_list)

    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    model_load_start = time.time()

    mode = mode.lower()
    if mode in [HYDRAGEN, HYDRAGEN_NOSHARED, NOATTENTION]:
        go_func = partial(
            go_hydragen,
            disable_hydragen=(mode == HYDRAGEN_NOSHARED),
            disable_attention=(mode == NOATTENTION),
        )
        model = HydragenLlamaForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="cuda"
        )
    elif mode == VLLM:
        go_func = partial(go_vllm)
        model = LLM(
            model=model_name,
            tokenizer=model_name,
            dtype="auto",
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            max_num_seqs=max(bs_list),
            max_paddings=max_paddings,
            tensor_parallel_size=tp,
            # default_shared_prefix_len=cfg.num_shared_list[0],
            # default_shared_groups=1,
            # enforce_eager=True,
        )
    else:
        raise ValueError(f"Invalid mode '{mode}'")

    model_load_end = time.time()
    print(f"Model loaded in {model_load_end - model_load_start:.2f} seconds")

    print("Starting benchmark")

    with tqdm(total=total) as pbar:
        for bs, num_shared, num_unique in product(
            bs_list,
            num_shared_list,
            num_unique_list,
        ):
            tag = f"b{bs}_s{num_shared}_u{num_unique}"
            outpath = outdir / mode / f"{tag}.yaml"

            if outpath.exists():
                continue

            print(f"Running {mode} {tag}...")

            times, warmup_times = go_func(
                model=model,
                tokenizer=tokenizer,
                bs=bs,
                num_shared=num_shared,
                num_unique=num_unique,
                num_iters=num_iters,
                num_warmup=num_warmup,
            )

            prefill_times = None
            prefill_warmup_times = None
            if calc_prefill:
                prefill_times, prefill_warmup_times = go_func(
                    model=model,
                    tokenizer=tokenizer,
                    bs=bs,
                    num_shared=num_shared,
                    num_unique=1,
                    num_iters=num_iters,
                    num_warmup=num_warmup,
                )

            result = SynthBenchmarkResult(
                bs=bs,
                num_shared=num_shared,
                num_unique=num_unique,
                times=times,
                prefill_times=prefill_times,
                warmup_times=warmup_times,
                prefill_warmup_times=prefill_warmup_times,
            )

            rstd = result.std() / result.mean()
            print(tag, result.mean(), result.std(), rstd)
            if rstd > 0.1:
                print(f"!!!!WARNING!!!!: high standard deviation for {tag}: {rstd}")

            outpath.parent.mkdir(parents=True, exist_ok=True)
            save_yaml(outpath, dataclass_to_dict(result))

            pbar.update(1)


if __name__ == "__main__":
    typer.run(sweep)
