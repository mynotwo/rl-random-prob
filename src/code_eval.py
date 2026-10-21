"""Code execution evaluation for RLVR coding experiments."""
from __future__ import annotations
import re
import subprocess
import tempfile
import os
from typing import Optional


CODE_PROMPT = (
    "Write a Python function to solve the following problem. "
    "Only output the function definition, no explanation.\n\n"
    "Problem: {question}\n"
    "Your code should pass this test:\n{test}\n\nSolution:"
)


def build_code_prompt(ex) -> str:
    if isinstance(ex, dict):
        question = ex.get("orig_question", "")
        tests = ex.get("test_list") or []
        test = tests[0] if tests else ""
    else:
        question, test = ex, ""
    return CODE_PROMPT.format(question=question, test=test)


def extract_code_block(text: str) -> str:
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    lines = text.strip().split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.rstrip()
        if stripped.startswith("def ") or stripped.startswith("import ") or stripped.startswith("from "):
            in_code = True
        if in_code:
            code_lines.append(line)
    if code_lines:
        return "\n".join(code_lines)
    return text.strip()


def execute_code_with_tests(code: str, test_list: list[str],
                            test_setup_code: str = "",
                            timeout: int = 10) -> tuple[bool, str]:
    full_code = ""
    if test_setup_code:
        full_code += test_setup_code + "\n"
    full_code += code + "\n\n"
    full_code += "\n".join(test_list) + "\n"
    full_code += 'print("ALL_TESTS_PASSED")\n'

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmppath = f.name

    try:
        result = subprocess.run(
            ["python3", tmppath],
            capture_output=True, timeout=timeout, text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        )
        passed = result.returncode == 0 and "ALL_TESTS_PASSED" in result.stdout
        err = result.stderr[:500] if result.stderr else ""
        return passed, err
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)[:200]
    finally:
        try:
            os.unlink(tmppath)
        except OSError:
            pass


def compute_code_rewards(texts: list[str], ex: dict,
                         reward_mode: str, random_reward_p: float,
                         rng) -> list[float]:
    import random as _random
    if reward_mode == "random":
        return [1.0 if rng.random() < random_reward_p else 0.0 for _ in texts]
    if reward_mode == "random_label":
        idx = rng.randint(0, len(texts) - 1)
        return [1.0 if i == idx else 0.0 for i in range(len(texts))]
    if reward_mode == "zero":
        return [0.0 for _ in texts]
    if reward_mode == "format":
        return [1.0 if ("def " in text or "```" in text) else 0.0 for text in texts]
    if reward_mode == "noisy_gt":
        test_list = ex.get("test_list", [])
        test_setup = ex.get("test_setup_code", "")
        rewards = []
        for text in texts:
            code = extract_code_block(text)
            passed, _ = execute_code_with_tests(code, test_list, test_setup, timeout=10)
            r = 1.0 if passed else 0.0
            if rng.random() < random_reward_p:
                r = 1.0 - r
            rewards.append(r)
        return rewards
    # ground_truth: execute code
    test_list = ex.get("test_list", [])
    test_setup = ex.get("test_setup_code", "")
    rewards = []
    for text in texts:
        code = extract_code_block(text)
        passed, _ = execute_code_with_tests(code, test_list, test_setup, timeout=10)
        rewards.append(1.0 if passed else 0.0)
    return rewards


def evaluate_code(texts: list[str], examples: list[dict]) -> dict:
    """Evaluate generated code against test cases."""
    passed = 0
    extracted = 0
    has_func = 0
    total = len(texts)

    eval_examples = []
    eval_outcomes = []

    for text, ex in zip(texts, examples):
        code = extract_code_block(text)
        has_code = bool(code and ("def " in code or "=" in code))
        extracted += int(has_code)
        has_func += int("def " in text)

        test_list = ex.get("test_list", [])
        test_setup = ex.get("test_setup_code", "")
        ok, err = execute_code_with_tests(code, test_list, test_setup, timeout=10)
        passed += int(ok)

        if len(eval_examples) < 5:
            eval_examples.append({
                "question": ex.get("orig_question", "")[:200],
                "generated_code": code[:300],
                "passed": ok,
                "error": err[:100] if not ok else "",
            })
        eval_outcomes.append("pass" if ok else "fail")

    n = max(1, total)
    return {
        "accuracy": passed / n,
        "strict_accuracy": passed / n,
        "extract_rate": extracted / n,
        "strict_extract_rate": extracted / n,
        "format_rate": has_func / n,
        "response_chars_mean": sum(len(t) for t in texts) / n,
        "response_tokens_mean": sum(len(t.split()) for t in texts) / n,
        "unique_token_ratio_mean": 0.0,
        "repeated_bigram_mean": 0.0,
        "code_like_rate": has_func / n,
        "examples": eval_examples,
        "outcomes": eval_outcomes[:20],
    }
