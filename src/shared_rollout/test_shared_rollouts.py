#!/usr/bin/env python3
"""CPU acceptance checks for the shared-rollout trainer (spec section 8, items 2-7 and the rerun-determinism that replaces item 8).
A 2-layer random Llama stands in for the 7B model: no download, no GPU, a few seconds.

    $PROBE_PYTHON test_shared_rollouts.py
"""
import copy, math, unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

import sr_common as C
import train_shared_rollouts as T

PAD, EOS = 63, 62


def tiny():
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                                        num_key_value_heads=2, max_position_embeddings=64, pad_token_id=PAD, eos_token_id=EOS))


def group(rewards=(1, 0, 0, 1)):
    resp = [[5, 6, 7, EOS], [8, 9], [10, 11, 12, 13, 14], [15, PAD, 16]]          # unequal lengths, one EOS, one pad id inside a response
    g = dict(step=1, train_item_id="x", prompt_input_ids=[1, 2, 3], response_input_ids=resp,
             response_loss_masks=[[int(t != PAD) for t in r] for r in resp], rewards=list(rewards),
             advantages=C.advantages(torch.tensor(rewards)).tolist())
    g["group_hash"] = C.group_hash(g)
    return g


class Spy(dict):
    def __init__(self, *a):
        super().__init__(*a); self.read = set()

    def __getitem__(self, k):
        self.read.add(k); return super().__getitem__(k)


class Tests(unittest.TestCase):
    def test_loss_by_hand(self):                       # shift, prompt mask, padding, EOS convention, token-mean denominator
        g = group(); seq, mask, adv = C.group_tensors(g, PAD, "cpu")
        self.assertEqual(seq.shape, (4, 8)); self.assertEqual(int(mask.sum()), 4 + 2 + 5 + 2)      # EOS counted, the in-response pad id not
        self.assertFalse(mask[:, :2].any())            # targets 1,2 are prompt tokens; target index 2 is the first response token
        self.assertTrue(mask[:, 2].all())
        torch.manual_seed(1); logits = torch.randn(4, 8, 64)
        loss, _ = C.offpolicy_loss(logits, seq, mask, adv)
        lp = torch.log_softmax(logits.double(), -1); tot = n = 0.0
        for i, r in enumerate(g["response_input_ids"]):
            for j, t in enumerate(r):
                if t != PAD:
                    tot += g["advantages"][i] * float(lp[i, 3 + j - 1, t]); n += 1
        self.assertAlmostEqual(float(loss), -tot / n, places=5)

    def test_advantages(self):
        self.assertTrue(torch.allclose(C.advantages(torch.tensor([1, 0, 0, 1])), torch.tensor([1., -1., -1., 1.]), atol=1e-4))
        self.assertEqual(C.advantages(torch.tensor([1, 1, 1, 1])).abs().sum().item(), 0.0)          # identical rewards: all zero, kept
        a, b = C.reward_table(51001), C.reward_table(51001)
        self.assertTrue(torch.equal(a, b)); self.assertFalse(torch.equal(a, C.reward_table(51002))); self.assertEqual(tuple(a.shape), (500, 4))

    def test_gradient_flips_with_rewards(self):        # same parameters, same batch, rewards 0<->1: A and the uncapped gradient change sign
        model, lora, _ = T.attach_lora(tiny(), 1)
        with torch.no_grad():
            for n, p in lora.items():
                if "lora_B" in n:
                    p.normal_(0, 0.02)                 # B=0 would make the A-gradients vanish
        grads = []
        for rw in ((1, 0, 0, 1), (0, 1, 1, 0)):
            seq, mask, adv = C.group_tensors(group(rw), PAD, "cpu")
            model.zero_grad(); C.offpolicy_loss(model(seq).logits, seq, mask, adv)[0].backward()
            grads.append([p.grad.clone() for p in lora.values()])
        self.assertTrue(all(torch.allclose(x, -y, atol=1e-6) for x, y in zip(*grads)))
        self.assertGreater(sum(float(x.abs().sum()) for x in grads[0]), 0)

    def test_zero_signal(self):
        model, lora, _ = T.attach_lora(tiny(), 1); opt = T.make_optimizer(lora)
        self.assertEqual(opt.param_groups[0]["weight_decay"], 0.0)
        before = C.tensors_hash(lora)
        seq, mask, adv = C.group_tensors(group((1, 1, 1, 1)), PAD, "cpu")
        rec = T.update(model, opt, lora, seq, mask, adv)
        self.assertEqual(rec["grad_norm"], 0.0); self.assertEqual(C.tensors_hash(lora), before)      # fresh optimizer + A=0: nothing moves
        T.update(model, opt, lora, *C.group_tensors(group((1, 0, 0, 1)), PAD, "cpu"))               # builds Adam momentum
        moved = C.tensors_hash(lora); steps = float(opt.state[next(iter(lora.values()))]["step"])   # copy: the state tensor is updated in place
        T.update(model, opt, lora, seq, mask, adv)                                                   # A=0 again: the step is still taken, not skipped
        self.assertEqual(float(opt.state[next(iter(lora.values()))]["step"]), steps + 1)
        self.assertNotEqual(C.tensors_hash(lora), moved)                                             # momentum moves the weights; that is expected

    def test_only_lora_moves_and_no_hidden_term(self):
        base = tiny(); ref = copy.deepcopy(base.state_dict())
        model, lora, _ = T.attach_lora(base, 1); opt = T.make_optimizer(lora)
        self.assertTrue(all("lora_" in n for n, p in model.named_parameters() if p.requires_grad))
        seq, mask, adv = C.group_tensors(group(), PAD, "cpu")
        with torch.no_grad():
            lp = torch.log_softmax(model(seq).logits[:, :-1].float(), -1).gather(-1, seq[:, 1:, None]).squeeze(-1)
        expect = -float((adv[:, None] * lp)[mask].mean())
        rec = T.update(model, opt, lora, seq, mask, adv)
        self.assertAlmostEqual(rec["loss"], expect, places=5)                                        # the loss IS the formula: no KL / entropy / SFT term
        for _ in range(3):
            T.update(model, opt, lora, seq, mask, adv)
        sd = model.state_dict()
        for k, v in ref.items():
            hit = [n for n in sd if n.endswith(k.replace(".weight", ".base_layer.weight")) or n.endswith(k)]
            self.assertTrue(any(torch.equal(sd[n], v) for n in hit), k)                              # every base tensor unchanged

    def test_training_reads_no_gold(self):
        g = Spy(group()); g["correct"] = [True] * 4; g["gold"] = 1.0
        C.group_tensors(g, PAD, "cpu")
        self.assertTrue(g.read <= T.USED_KEYS and not g.read & {"correct", "gold", "response_text", "prompt_text"})
        h = group(); h2 = dict(h, correct=[False] * 4)
        self.assertEqual(C.group_hash(h), C.group_hash(h2))                                          # a correctness sidecar cannot change the cache key

    def test_shared_adapter_start_and_rerun_determinism(self):
        finals = []
        for _ in range(2):
            model, lora, h = T.attach_lora(tiny(), 7); opt = T.make_optimizer(lora)
            self.assertTrue(all(float(p.abs().sum()) == 0 for n, p in lora.items() if "lora_B" in n))
            for rw in ((1, 0, 0, 1), (0, 0, 1, 1), (1, 1, 1, 1), (0, 1, 0, 0), (1, 0, 1, 1)):
                T.update(model, opt, lora, *C.group_tensors(group(rw), PAD, "cpu"))
            finals.append((h, C.tensors_hash(lora)))
        self.assertEqual(finals[0], finals[1])                                                       # same start, same 5 steps -> same adapter
        self.assertNotEqual(T.attach_lora(tiny(), 8)[2], finals[0][0])
        a = next(v for k, v in C.lora_init_tensors({"x.lora_A.w": (8, 32)}, 7).items())
        self.assertLessEqual(float(a.abs().max()), 1 / math.sqrt(32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
