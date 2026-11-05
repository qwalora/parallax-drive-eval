#!/usr/bin/env python3
"""Run one open-weight vision-language model over a JSONL file of prepared trials.

Usage: python -I run_model.py --model REPO --revision SHA --trials trials.jsonl --out raw.jsonl
       [--images-root DIR] [--max-new-tokens N] [--shard I/N]

Each trial line: {"trial_id", "images": [relative paths], "system": str, "user": str}.
Each output line: {"trial_id", "model", "revision", "output", "latency_s", "input_tokens",
                   "output_tokens", "error"}.
Decoding is greedy (do_sample=False), so a rerun on the same software and hardware gives the
same text. The script only produces text. Nothing it outputs is connected to any vehicle,
navigation service or actuator.
"""
import argparse
import json
import os
import platform
import resource
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--trials", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images-root", default=".")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import transformers

    torch.manual_seed(0)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[a.dtype]
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForImageTextToText.from_pretrained(a.model, revision=a.revision, dtype=dtype)
    model.eval()
    load_s = time.time() - t0

    i, n = (int(x) for x in a.shard.split("/"))
    trials = [json.loads(l) for l in open(a.trials)]
    trials = [t for k, t in enumerate(trials) if k % n == i]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["trial_id"] for l in open(a.out) if l.strip()}
    out = open(a.out, "a")
    env = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
           "machine": platform.machine(), "cpus": os.cpu_count(), "dtype": a.dtype, "load_s": round(load_s, 1)}
    print("ENV " + json.dumps(env), flush=True)
    for t in trials:
        if t["trial_id"] in done:
            continue
        rec = {"trial_id": t["trial_id"], "model": a.model, "revision": a.revision, "output": None,
               "latency_s": None, "input_tokens": None, "output_tokens": None, "error": None}
        try:
            turns = t.get("turns") or [{"images": t["images"], "user": t["user"]}]
            msgs = [{"role": "system", "content": [{"type": "text", "text": t["system"]}]}]
            outs, lat, itok, otok = [], 0.0, [], []
            for turn in turns:
                content = [{"type": "image", "image": Image.open(os.path.join(a.images_root, p)).convert("RGB")}
                           for p in turn["images"]]
                content.append({"type": "text", "text": turn["user"]})
                msgs.append({"role": "user", "content": content})
                inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                                  return_dict=True, return_tensors="pt")
                s = time.time()
                with torch.inference_mode():
                    gen = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
                new = gen[0, inputs["input_ids"].shape[1]:]
                text = proc.decode(new, skip_special_tokens=True)
                lat += time.time() - s
                outs.append(text)
                itok.append(int(inputs["input_ids"].shape[1]))
                otok.append(int(new.shape[0]))
                msgs.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
            rec["output"] = outs[0] if len(outs) == 1 else outs
            rec["latency_s"] = round(lat, 3)
            rec["input_tokens"] = itok[0] if len(itok) == 1 else itok
            rec["output_tokens"] = otok[0] if len(otok) == 1 else otok
        except Exception as e:  # recorded, never silently dropped
            rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:400])
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print("%s %.1fs %s" % (t["trial_id"], rec["latency_s"] or -1, str(rec["output"] or rec["error"] or "")[:120].replace("\n", " ")), flush=True)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print("PEAK_RSS_MB %.0f" % peak, flush=True)
    with open(a.out + ".env.json", "w") as f:
        env["peak_rss_mb"] = round(peak)
        json.dump(env, f, indent=2)


if __name__ == "__main__":
    main()
