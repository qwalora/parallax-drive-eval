#!/usr/bin/env python3
"""Run trials against a local llama-server (llama.cpp) hosting an official GGUF build of a VLM.

Usage: python -I run_llamacpp.py --url http://127.0.0.1:8080 --model-label LABEL --revision SHA
       --trials trials.jsonl --out raw.jsonl [--images-root DIR] [--shard I/N] [--max-new-tokens N]

Same trial and output format as run_model.py. Decoding is greedy: temperature 0, top_k 1, a fixed
seed, and no prompt cache reuse across trials. The server listens on localhost only. Output text is
scored afterwards and is never connected to any vehicle, service or actuator.
"""
import argparse
import base64
import json
import os
import time
import urllib.request


def post(url, body, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def img_part(root, p):
    b = base64.b64encode(open(os.path.join(root, p), "rb").read()).decode()
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--trials", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images-root", default=".")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    a = ap.parse_args()
    i, n = (int(x) for x in a.shard.split("/"))
    trials = [json.loads(l) for k, l in enumerate(open(a.trials)) if k % n == i]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["trial_id"] for l in open(a.out) if l.strip()}
    out = open(a.out, "a")
    for t in trials:
        if t["trial_id"] in done:
            continue
        rec = {"trial_id": t["trial_id"], "model": a.model_label, "revision": a.revision, "output": None,
               "latency_s": None, "input_tokens": None, "output_tokens": None, "error": None}
        try:
            turns = t.get("turns") or [{"images": t["images"], "user": t["user"]}]
            msgs = [{"role": "system", "content": t["system"]}]
            outs, lat, itok, otok = [], 0.0, [], []
            for turn in turns:
                msgs.append({"role": "user", "content": [img_part(a.images_root, p) for p in turn["images"]]
                             + [{"type": "text", "text": turn["user"]}]})
                s = time.time()
                r = post(a.url + "/v1/chat/completions", {
                    "messages": msgs, "temperature": 0, "top_k": 1, "top_p": 1.0, "seed": 0,
                    "max_tokens": a.max_new_tokens, "cache_prompt": False, "stream": False})
                lat += time.time() - s
                text = r["choices"][0]["message"]["content"]
                outs.append(text)
                u = r.get("usage", {})
                itok.append(u.get("prompt_tokens"))
                otok.append(u.get("completion_tokens"))
                msgs.append({"role": "assistant", "content": text})
            rec["output"] = outs[0] if len(outs) == 1 else outs
            rec["latency_s"] = round(lat, 3)
            rec["input_tokens"] = itok[0] if len(itok) == 1 else itok
            rec["output_tokens"] = otok[0] if len(otok) == 1 else otok
        except Exception as e:  # recorded, never silently dropped
            rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:400])
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print("%s %.1fs %s" % (t["trial_id"], rec["latency_s"] or -1, str(rec["output"] or rec["error"] or "")[:110].replace("\n", " ")), flush=True)


if __name__ == "__main__":
    main()
