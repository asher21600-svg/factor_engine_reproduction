#!/usr/bin/env python
"""Verify a LIVE LLM (e.g. Kimi) can reach the evolution code path.

Unlike a plain `curl` reachability check, this exercises the engine's real
LLM call path via fe.evolution.macro._call_llm (honors FE_LLM_PROXY, DNS
fallback, model temperature rules, and provider env).  Set
FE_LLM_CHECK_MUTATION=1 to additionally ask for a full SEARCH/REPLACE factor
mutation and compile it.

Exit codes:  0 = reachable via the engine's LLM call path
             1 = unreachable / auth / API error
             2 = reachable, but optional mutation smoke was not parseable

Usage (on a network that can reach the endpoint):
    export FE_LLM_PROVIDER=kimi-cn
    export FE_LLM_BASE_URL=https://api.moonshot.cn/v1
    export FE_LLM_API_KEY=sk-...        FE_LLM_MODEL=kimi-k2.6
    unset FE_LLM_PROXY                  # direct HTTPS is preferred in China
    python scripts/check_llm.py
"""
import _bootstrap  # noqa: F401  (puts repo root on sys.path)

import os
import sys
import time
from urllib.parse import urlparse
import socket

from fe.factors import SEED_SRC
from fe.factors.contract import compile_factor, FactorRunError
from fe.evolution.tree import ProgramNode
from fe.evolution import macro


def _uses_remote_dns_proxy(proxy: str) -> bool:
    return proxy.lower().startswith(("socks5h://", "http://", "https://"))


def _print_route_hint(base: str, proxy: str) -> None:
    host = urlparse(base).hostname or base
    print(f"[FAIL] DNS cannot resolve {host!r} from this shell.")
    if proxy:
        print(f"       FE_LLM_PROXY is set to {proxy!r}. If you are in China, unset it and use direct HTTPS.")
        print("       Run: FE_LLM_PROXY= ./run_local.sh")
    else:
        print("       No FE_LLM_PROXY is configured; this is the intended China-direct HTTPS path.")
        print("       Check local DNS/network access to https://api.moonshot.cn/v1.")


def _preflight_dns(bases: list[str], proxy: str) -> None:
    if proxy and _uses_remote_dns_proxy(proxy):
        return
    for base in bases:
        host = urlparse(base).hostname
        if not host:
            continue
        ips, source = (None, "")
        for attempt in range(3):           # retry: DNS via the VPN resolver fluctuates
            ips, source = macro.resolve_llm_host(host, 443)
            if ips:
                break
            time.sleep(1.5)
        if ips:
            print(f"dns      = {source} ({host} -> {ips[0]})")
        else:
            # Do NOT hard-exit: the live call below retries with backoff and has a
            # DNS fallback. A transient preflight miss must not block the run.
            print(f"dns      = preflight could not resolve {host} after 3 tries "
                  "(continuing; the live call retries + uses the DNS fallback).")


def _preflight_proxy(proxy: str) -> None:
    if not proxy:
        return
    p = urlparse(proxy)
    if not p.hostname or not p.port:
        print(f"[FAIL] FE_LLM_PROXY is not a valid proxy URL: {proxy!r}")
        print("       In China, prefer direct HTTPS: FE_LLM_PROXY= ./run_local.sh")
        print("       If you truly need a proxy, use a URL like https://127.0.0.1:7890")
        sys.exit(1)
    try:
        with socket.create_connection((p.hostname, p.port), timeout=2):
            return
    except OSError as e:
        print(f"[FAIL] Cannot connect to FE_LLM_PROXY {proxy!r}: {e}")
        print("       Since you are in China, you probably do not need this proxy.")
        print("       Run direct HTTPS instead: FE_LLM_PROXY= ./run_local.sh")
        print("       If you truly need a proxy, start it, enable its local listen port, or use the correct port.")
        print("       Quick checks:")
        print(f"         nc -vz {p.hostname} {p.port}")
        print("         lsof -nP -iTCP -sTCP:LISTEN | grep -E '1080|7890|7891|6152'")
        print("       HTTPS proxy example:")
        print("         FE_LLM_PROXY=https://127.0.0.1:7890 ./run_local.sh")
        sys.exit(1)


def main():
    cfg = macro.describe_llm_config()
    bases = cfg.get("base_urls", [])
    print(f"provider = {cfg.get('provider')}")
    print(f"endpoint = {', '.join(bases) if bases else '(none -> Anthropic/none)'}")
    print(f"model    = {cfg.get('model')}")
    print(f"proxy    = {cfg.get('proxy') or 'none'}")
    print(f"thinking = {cfg.get('thinking') or 'default'}")
    print(f"api key  = {'set' if cfg.get('api_key_present') or cfg.get('anthropic_key_present') else 'MISSING'}")
    print(f"dns fb   = {'on' if cfg.get('dns_fallback') else 'off'}")
    if bases and not cfg.get("api_key_present"):
        print("[FAIL] FE_LLM_API_KEY / MOONSHOT_API_KEY / KIMI_API_KEY not set")
        sys.exit(1)
    if not bases and not cfg.get("anthropic_key_present"):
        print("[FAIL] no live LLM configured; set FE_LLM_PROVIDER=kimi-cn and FE_LLM_API_KEY")
        sys.exit(1)
    _preflight_proxy(cfg.get("proxy") or "")
    _preflight_dns(bases, cfg.get("proxy") or "")

    # 1) raw connectivity + a trivial completion
    t0 = time.time()
    try:
        txt = macro._call_llm("You are a quantitative researcher.",
                              "Reply with the single word: PONG.",
                              model=cfg.get("model"), max_tokens=16)
    except Exception as e:  # noqa: BLE001
        err = str(e)
        print(f"[FAIL] LLM call errored after {time.time()-t0:.1f}s: "
              f"{type(e).__name__}: {err[:240]}")
        route_words = ("nodename nor servname", "Name or service", "handshake operation timed out",
                       "timed out", "Connection refused", "Network is unreachable")
        if any(w in err for w in route_words):
            print("       This looks like endpoint routing, not factor-engine logic.")
            print("       In China, use direct HTTPS to api.moonshot.cn:")
            print("       FE_LLM_PROXY= FE_LLM_BASE_URL=https://api.moonshot.cn/v1 ./run_local.sh")
            print("       If your environment forces a bad system proxy, leave FE_LLM_USE_SYSTEM_PROXY unset or 0.")
        print("       -> check endpoint reachability / proxy / key / model name.")
        sys.exit(1)
    print(f"\n[{time.time()-t0:.1f}s] raw completion: {str(txt)[:80]!r}")
    if not txt:
        print("[FAIL] empty response"); sys.exit(1)
    if str(txt).strip():
        used = macro.describe_llm_config().get("last_success", {})
        if used:
            print(f"provider used : {used.get('provider')} {used.get('model')} @ {used.get('base_url')}")
    if not (os.environ.get("FE_LLM_CHECK_MUTATION") in ("1", "true", "True")):
        print("\n[OK] LIVE LLM IS DRIVING — reachable through the evolution call path.")
        print("     Full mutation parsing will be handled inside the evolution loop.")
        sys.exit(0)

    # 2) full macro-mutation round-trip on the seed factor
    print("\nAsking the LLM for a real factor mutation (SEARCH/REPLACE diff)...")
    t0 = time.time()
    mut = macro.llm_mutation(ProgramNode(code=SEED_SRC), SEED_SRC,
                             "(no evolution history yet)",
                             metrics_desc="{dataset: smoke-test seed panel}",
                             model=cfg.get("model"))
    dt = time.time() - t0
    if mut is None:
        print(f"[WARN] ({dt:.1f}s) LLM is reachable but this smoke response had no valid SEARCH/REPLACE diff.")
        print("       The evolution engine will retry live calls and fall back only for unusable responses.")
        sys.exit(2)
    if getattr(mut, "tag", "") == "llm_error":
        print(f"[FAIL] {mut.idea}"); sys.exit(1)
    try:
        compile_factor(mut.code)
    except FactorRunError as e:
        print(f"[WARN] ({dt:.1f}s) parsed a diff but it does not compile: {e}")
        sys.exit(2)

    print(f"[{dt:.1f}s] LLM-proposed mutation parsed AND compiles  (tag={mut.tag})")
    print(f"  idea : {mut.idea[:200]}")
    print(f"  param ranges : {mut.param_space}")
    used = macro.describe_llm_config().get("last_success", {})
    if used:
        print(f"  provider used : {used.get('provider')} {used.get('model')} @ {used.get('base_url')}")
    print("\n[OK] LIVE LLM IS DRIVING — reachable, parsed, compilable.")
    print("     Now run:  FE_REQUIRE_LLM=1 python scripts/02_run_evolution.py "
          "--panel csi300 --iterations 200 --use-llm")


if __name__ == "__main__":
    main()
