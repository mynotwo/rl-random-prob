"""What the checkpoint_curve, cross_model and input_controls job generators share: the trainer command of the
fixed 500-step protocol, file hashing and the jobs.jsonl writer.

shared_rollout/gen_jobs.py does not import this module: it snapshots its own sources into every run root.
"""
import hashlib, json, os, pathlib, shlex

PY = os.environ.get("PROBE_PYTHON", "python3")
KL = {"random": "0.01", "ground_truth": "0"}


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def rl_run(trainer, dataset, odir, model, rev, reward, seed, suffix, steps=500):
    """(shell command, result file) of one RL run; the file name is the one the trainer derives from its arguments."""
    cmd = [PY, str(trainer), "--dataset_path", str(dataset), "--output_dir", odir, "--base_model", model, "--revision", rev,
           "--train_mode", "original", "--reward_mode", reward, "--seed", str(seed), "--split_seed", "777",
           "--train_steps", str(steps), "--group_size", "4", "--max_new_tokens", "96", "--learning_rate", "3e-5",
           "--kl_coeff", KL[reward], "--eval_every", str(min(100, steps)), "--eval_limit", "200", "--eval_batch_size", "8",
           "--lora_r", "8", "--lora_alpha", "16", "--lora_dropout", "0", "--run_suffix", suffix, "--save_adapter"]
    return shlex.join(cmd), f"{odir}/rlvr_{model.split('/')[-1]}_{rev}_original_{reward}_seed{seed}_{suffix}.jsonl"


def write_jobs(path, jobs):
    with open(path, "w") as f:
        for i, j in enumerate(jobs):
            f.write(json.dumps({"idx": i, **j}) + "\n")
