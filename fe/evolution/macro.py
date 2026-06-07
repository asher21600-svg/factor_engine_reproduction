"""Macro-level idea generation — code-logic mutation (paper §4.2, "Idea Generation").

Two interchangeable backends:

  * DETERMINISTIC transform library — a small set of code rewrites (turnover
    weighting, mid-price centering, rank-normalization, EWM smoothing) applied
    as targeted SEARCH/REPLACE edits.  Guarantees the engine can improve a seed
    even with no API key, and makes runs fully reproducible.

  * Live LLM — builds the paper's system prompt (Listing 1.1) + chain-of-
    experience context (Listing 1.2), asks the configured provider for
    `###Code changes` as SEARCH/REPLACE diffs plus a `###Parameters` JSON of
    Bayesian search ranges, and applies them.  Kimi/Moonshot's China endpoint is
    first-class via FE_LLM_PROVIDER=kimi-cn or FE_LLM_BASE_URL=https://api.moonshot.cn/v1.

Both return a `Mutation(code, param_space, idea, summary, tag)`.
"""
from __future__ import annotations

import json
import os
import re
import socket
import struct
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from ..factors.contract import compile_factor, FactorRunError


@dataclass
class Mutation:
    code: str
    param_space: dict = field(default_factory=dict)
    idea: str = ""
    summary: str = ""
    tag: str = ""           # transform id (deterministic) or 'llm'


KIMI_CN_BASE_URL = "https://api.moonshot.cn/v1"
KIMI_AI_BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_KIMI_MODEL = "kimi-k2.6"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_DNS_SERVERS = ("223.5.5.5", "119.29.29.29", "114.114.114.114")
_LAST_LLM_INFO: dict = {}
_DNS_CACHE: dict[str, list[str]] = {}
_LLM_FAILURE_COUNT = 0


# Base search space for the seed family
BASE_PARAM_SPACE = {
    "w1": {"type": "float", "low": 0.0, "high": 1.0},
    "w2": {"type": "float", "low": 0.0, "high": 1.0},
    "w3": {"type": "float", "low": 0.0, "high": 1.0},
}


# --------------------------------------------------------------------------
# Deterministic transform library
# --------------------------------------------------------------------------
def _t_turnover(code):
    if "pl.col('volume') *" not in code:
        return None
    new = code.replace("pl.col('volume') *", "(pl.col('volume') * pl.col('close')) *")
    return Mutation(new, {}, "Weight each intraday pressure component by turnover "
                    "(volume x close) instead of raw share volume — capital-weighted signal.",
                    "volume -> turnover", "turnover")


def _t_midprice(code):
    anchor = "(pl.col('close') - pl.col('low'))"
    if anchor not in code:
        return None
    new = code.replace(anchor, "(pl.col('close') - (pl.col('high') + pl.col('low')) / 2.0)", 1)
    return Mutation(new, {}, "Measure close relative to the day's MID-price instead of "
                    "the low — a sign-symmetric, unbiased read of intraday pressure.",
                    "close-low -> close-mid", "midprice")


def _t_ranknorm(code):
    repls = []
    for s in ("sf1_expr", "sf2_expr", "sf3_expr"):
        zexpr = f"({s} - {s}.mean().over('datetime')) / ({s}.std(ddof=0).over('datetime') + EPSILON)"
        rexpr = f"({s}.rank(method='average').over('datetime') / ({s}.count().over('datetime') + 1) - 0.5)"
        if zexpr in code:
            repls.append((zexpr, rexpr))
    if not repls:
        return None
    new = code
    for a, b in repls:
        new = new.replace(a, b)
    return Mutation(new, {}, "Replace cross-sectional z-scoring of each component with "
                    "rank-normalization — robust to fat-tailed turnover outliers.",
                    "z-score -> rank-norm", "ranknorm")


def _t_ewm(code):
    final = "(w1 * pl.col('z1') + w2 * pl.col('z2') + w3 * pl.col('z3')).alias('Factor')"
    select = "df_tf = df_factor.select(['instrument', 'datetime', 'Factor'])"
    if final not in code or select not in code or "ewm_mean" in code:
        return None
    new = code.replace(final, "(w1 * pl.col('z1') + w2 * pl.col('z2') + w3 * pl.col('z3')).alias('raw_f')")
    smooth = (
        "df_factor = df_factor.sort(['instrument','datetime']).with_columns(\n"
        "        pl.col('raw_f').ewm_mean(span=smoothing_window, min_periods=1).over('instrument').alias('smoothed_f'))\n"
        "    df_factor = df_factor.with_columns(\n"
        "        ((pl.col('smoothed_f') - pl.col('smoothed_f').mean().over('datetime')) /\n"
        "         (pl.col('smoothed_f').std(ddof=0).over('datetime') + EPSILON)).alias('Factor'))\n"
        "    " + select)
    new = new.replace(select, smooth)
    # inject the param default at the top of the function body
    new = new.replace('EPSILON = parameters.get("epsilon", 1e-9)',
                      'EPSILON = parameters.get("epsilon", 1e-9)\n    smoothing_window = parameters.get("smoothing_window", 5)')
    return Mutation(new, {"smoothing_window": {"type": "int", "low": 2, "high": 20}},
                    "Add EWM temporal smoothing of the combined factor per instrument, "
                    "then re-standardize cross-sectionally — denoise a persistent signal.",
                    "+ EWM smoothing", "ewm")


def _t_volscale(code):
    """Risk-normalize the final factor by its trailing per-instrument volatility."""
    select = "df_tf = df_factor.select(['instrument', 'datetime', 'Factor'])"
    if select not in code or "rolling_std" in code:
        return None
    block = (
        "df_factor = df_factor.sort(['instrument','datetime']).with_columns(\n"
        "        (pl.col('Factor') / (pl.col('Factor').rolling_std(window_size=vol_window, min_periods=2).over('instrument') + EPSILON)).alias('Factor'))\n"
        "    " + select)
    new = code.replace(select, block)
    new = new.replace('EPSILON = parameters.get("epsilon", 1e-9)',
                      'EPSILON = parameters.get("epsilon", 1e-9)\n    vol_window = parameters.get("vol_window", 20)')
    return Mutation(new, {"vol_window": {"type": "int", "low": 5, "high": 40}},
                    "Risk-normalize the factor by its trailing per-instrument volatility "
                    "(scale by 1/rolling-std) — stabilizes exposure across regimes.",
                    "+ vol scaling", "volscale")


def _t_neg(code):
    """Sign flip — lets the search invert a reversal-type signal."""
    final = "(w1 * pl.col('z1') + w2 * pl.col('z2') + w3 * pl.col('z3'))"
    if final not in code or "ranknorm" in code or "ewm" in code:
        return None  # only applicable to the base z-score form (stable anchor)
    new = code.replace(final, "(-1.0) * " + final, 1)
    return Mutation(new, {}, "Invert the factor sign (capture reversal vs momentum).",
                    "sign flip", "neg")


def _make_proposal(name, src, pspace):
    """Wrap an offline LLM-reasoned full-program proposal as a fresh macro idea."""
    tag = f"proposal_{name}"

    def _fn(code):  # ignores current code — a proposal is a fresh factor family
        return Mutation(src, dict(pspace), f"[LLM idea: {name}]", f"propose {name}", tag)
    _fn.__name__ = f"_t_{tag}"
    return _fn


def _proposal_transforms():
    from ..factors.llm_proposals import PROPOSALS
    return [_make_proposal(n, s, p) for n, (s, p) in PROPOSALS.items()]


# 6 incremental code transforms + offline full-program proposals.  The live
# Kimi/Moonshot path below can add fresh dataset-conditioned mutations when
# credentials and network access are available.
DET_TRANSFORMS = ([_t_turnover, _t_midprice, _t_ranknorm, _t_ewm, _t_volscale, _t_neg]
                  + _proposal_transforms())


def deterministic_mutation(node, rng) -> Mutation | None:
    """Apply one not-yet-used transform/proposal whose code compiles."""
    order = list(DET_TRANSFORMS)
    rng.shuffle(order)
    for fn in order:
        tag = fn.__name__.replace("_t_", "")
        if tag in node.transforms:
            continue
        mut = fn(node.code)
        if mut is None:
            continue
        try:
            compile_factor(mut.code)        # syntactic safety check
        except FactorRunError:
            continue
        return mut
    return None


# --------------------------------------------------------------------------
# Live LLM backend — paper Listing 1.1 / 1.2
# --------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are one of the most authoritative quantitative researchers at a top "
    "Wall Street hedge fund. You design and implement new alpha factors as "
    "executable Polars programs to maximize predictive metrics (IC, ICIR, Rank "
    "IC, Rank ICIR aggregated across 1/3/5/10-day horizons) while strictly "
    "avoiding any look-ahead bias or data leakage. Factors take a pandas/Polars "
    "OHLCV frame and a `parameters` dict and return a Polars DataFrame with "
    "columns [instrument, datetime, Factor]. The primary target is Chinese "
    "A-share cross-sectional prediction on CSI300/CSI500-style daily OHLCV "
    "panels, where liquidity, turnover, short-horizon reversal, volatility "
    "state, and price-volume confirmation often matter. Improve the CURRENT "
    "program with targeted edits. Prefer exact SEARCH/REPLACE diffs. If exact "
    "matching is uncertain, provide a complete replacement `def factor(...)` "
    "program instead."
)

RESPONSE_FORMAT = r"""
Respond in one of these two formats.

Preferred format:

###Analyse: <domain insight from comparing current vs original program and the lessons from the evolution history>

###IDEA: <one concrete idea to raise the metrics; focus on both factor logic and tunable parameters>

###Code changes:
<<<<<<< SEARCH
# exact snippet from the CURRENT program to replace (must match character-for-character)
=======
# replacement snippet
>>>>>>> REPLACE

(You may give multiple SEARCH/REPLACE blocks. Each SEARCH must match the CURRENT program exactly.)

Fallback format if exact SEARCH snippets are uncertain:

###Analyse: <domain insight>

###IDEA: <one concrete idea>

###Full code:
```python
def factor(pricing_data, parameters):
    ...
```

###Parameters: a JSON object of Bayesian-search ranges. RULES (enforced):
  - Declare a range for EVERY tunable the code reads via parameters.get(...) — any
    numeric knob left undeclared will be auto-added to the search (do not rely on
    hardcoded defaults; they will be tuned regardless).
  - Keep ONE mutation theme per response; do not combine unrelated edits.
  - No look-ahead: only use information available at or before each datetime.
  - Add a one-line rationale tied to CSI300/CSI500 microstructure in ###IDEA.
Example:
{"w3": {"type": "float", "low": 0.3, "high": 0.9}, "smoothing_window": {"type": "int", "low": 3, "high": 20}}
"""


def _parse_search_replace(text: str, code: str):
    """Apply SEARCH/REPLACE blocks; return new code (or None if none applied)."""
    blocks = re.findall(r"<{5,}\s*SEARCH\s*(.*?)={5,}\s*(.*?)>{5,}\s*REPLACE",
                        text, flags=re.DOTALL)
    if not blocks:
        return None
    new = code
    applied = 0
    for search, replace in blocks:
        s = search.strip("\n")
        r = replace.strip("\n")
        if s and s in new:
            new = new.replace(s, r, 1)
            applied += 1
    return new if applied else None


def _parse_full_code(text: str) -> str | None:
    """Extract a full replacement factor program from an LLM response."""
    section = _section(text, "Full code")
    candidates = []
    if section:
        candidates.append(section)
    candidates += re.findall(r"```(?:python)?\s*(def\s+factor\(.*?)(?:```|\Z)",
                             text, flags=re.DOTALL)
    if "def factor(" in text:
        candidates.append(text[text.find("def factor("):])
    for cand in candidates:
        code = cand.strip()
        code = re.sub(r"^```(?:python)?\s*", "", code)
        code = re.sub(r"\s*```$", "", code).strip()
        if "def factor(" not in code:
            continue
        # Drop trailing markdown/explanation after the function when present.
        trailing = re.search(r"\n###|\n(?:Analysis|Explanation|Parameters):", code)
        if trailing:
            code = code[:trailing.start()].rstrip()
        return code
    return None


def _infer_param_spec(default_literal: str):
    """Infer a Bayesian-search spec from a parameters.get default literal."""
    s = default_literal.strip()
    try:
        if re.fullmatch(r"[-+]?\d+", s):          # int default
            d = int(s)
            lo, hi = (max(2, d // 2), max(d * 2, d + 4)) if d > 0 else (2, 30)
            return {"type": "int", "low": int(lo), "high": int(hi)}
        v = float(s)                               # float default
        if v == 0:
            return {"type": "float", "low": -0.5, "high": 0.5}
        lo, hi = sorted((v * 0.3, v * 1.8) if v > 0 else (v * 1.8, v * 0.3))
        return {"type": "float", "low": float(lo), "high": float(hi)}
    except ValueError:
        return None                                # non-numeric (e.g. string) — skip


def _auto_declare_params(code: str, pspace: dict) -> dict:
    """V3 #6: every numeric `parameters.get(name, default)` the code reads must be
    tunable. Undeclared numeric knobs are auto-added with an inferred range so they
    enter the Bayesian micro-search instead of silently staying at a hardcoded value."""
    pspace = dict(pspace or {})
    for name, default in re.findall(r'parameters\.get\(\s*["\']([^"\']+)["\']\s*,\s*([^)]+)\)', code or ""):
        if name in pspace or name == "epsilon":
            continue
        spec = _infer_param_spec(default)
        if spec:
            pspace[name] = spec
    return pspace


def _parse_params(text: str) -> dict:
    m = re.search(r"###Parameters:\s*(\{.*\})", text, flags=re.DOTALL)
    if not m:
        return {}
    blob = m.group(1)
    # take the first balanced JSON object
    depth, end = 0, None
    for i, ch in enumerate(blob):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    try:
        spec = json.loads(blob[:end])
        # keep only well-formed numeric specs
        clean = {}
        for k, v in spec.items():
            if isinstance(v, dict) and v.get("type") in ("float", "int") and "low" in v and "high" in v:
                clean[k] = v
        return clean
    except Exception:  # noqa: BLE001
        return {}


def _section(text: str, name: str) -> str:
    m = re.search(rf"###{re.escape(name)}:\s*(.*?)(?=\n###|\Z)", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _dump_llm_failure(text: str, reason: str) -> None:
    """Persist a small raw response sample for prompt/parser debugging."""
    if os.environ.get("FE_LLM_SAVE_FAILURES", "1") in ("0", "false", "False"):
        return
    try:
        root = Path(__file__).resolve().parents[2]
        out = root / "outputs" / "llm_failures"
        out.mkdir(parents=True, exist_ok=True)
        global _LLM_FAILURE_COUNT
        _LLM_FAILURE_COUNT += 1
        path = out / f"llm_failure_{_LLM_FAILURE_COUNT:03d}.txt"
        path.write_text(f"reason: {reason}\n\n{text[:12000]}", encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _csv_env(name: str) -> list[str]:
    value = os.environ.get(name, "")
    return [p.strip() for p in re.split(r"[,\s]+", value) if p.strip()]


def _dedupe(values: list[str]) -> list[str]:
    out, seen = [], set()
    for v in values:
        key = v.rstrip("/")
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _llm_provider() -> str:
    return os.environ.get("FE_LLM_PROVIDER", "").strip().lower()


def _configured_openai_bases() -> list[str]:
    """Return OpenAI-compatible base URLs in retry order."""
    bases = []
    bases += _csv_env("FE_LLM_BASE_URLS")
    bases += _csv_env("FE_LLM_BASE_URL")
    bases += _csv_env("MOONSHOT_BASE_URL")
    bases += _csv_env("OPENAI_BASE_URL")
    bases += _csv_env("FE_LLM_FALLBACK_BASE_URLS")
    provider = _llm_provider()
    has_moonshot_key = bool(os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY"))
    if not bases and (has_moonshot_key or provider in {"kimi", "moonshot", "moonshot-cn", "kimi-cn"}):
        bases.append(KIMI_CN_BASE_URL)
    if not bases and provider in {"moonshot-ai", "kimi-ai"}:
        bases.append(KIMI_AI_BASE_URL)
    if provider in {"kimi", "moonshot"}:
        bases += [KIMI_CN_BASE_URL, KIMI_AI_BASE_URL]
    return _dedupe(bases)


def _resolve_api_key() -> str:
    return (os.environ.get("FE_LLM_API_KEY")
            or os.environ.get("MOONSHOT_API_KEY")
            or os.environ.get("KIMI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "")


def _resolve_model(model=None, *, anthropic_provider=False) -> str:
    if model:
        return model
    if os.environ.get("FE_LLM_MODEL"):
        return os.environ["FE_LLM_MODEL"]
    if os.environ.get("MOONSHOT_MODEL"):
        return os.environ["MOONSHOT_MODEL"]
    if anthropic_provider:
        return os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
    return DEFAULT_KIMI_MODEL


def _proxy_url() -> str:
    if os.environ.get("FE_LLM_USE_SYSTEM_PROXY") in ("1", "true", "True"):
        return (os.environ.get("FE_LLM_PROXY") or os.environ.get("HTTPS_PROXY")
                or os.environ.get("https_proxy") or "")
    return os.environ.get("FE_LLM_PROXY", "")


def _dns_fallback_enabled() -> bool:
    return os.environ.get("FE_LLM_DNS_FALLBACK", "1") not in ("0", "false", "False")


def _dns_servers() -> list[str]:
    servers = _csv_env("FE_LLM_DNS_SERVERS")
    return servers or list(DEFAULT_DNS_SERVERS)


def _encode_dns_name(host: str) -> bytes:
    return b"".join(bytes([len(part)]) + part.encode("ascii")
                    for part in host.rstrip(".").split(".")) + b"\0"


def _skip_dns_name(buf: bytes, offset: int) -> int:
    while offset < len(buf):
        ln = buf[offset]
        if ln & 0xC0 == 0xC0:
            return offset + 2
        if ln == 0:
            return offset + 1
        offset += 1 + ln
    raise ValueError("truncated DNS name")


def _query_dns_a(host: str, server: str, timeout: float = 2.0) -> list[str]:
    qid = struct.unpack("!H", os.urandom(2))[0]
    packet = struct.pack("!HHHHHH", qid, 0x0100, 1, 0, 0, 0)
    packet += _encode_dns_name(host) + struct.pack("!HH", 1, 1)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(timeout)
        s.sendto(packet, (server, 53))
        data, _ = s.recvfrom(512)
    if len(data) < 12:
        return []
    rid, _flags, qdcount, ancount, _nscount, _arcount = struct.unpack("!HHHHHH", data[:12])
    if rid != qid:
        return []
    off = 12
    for _ in range(qdcount):
        off = _skip_dns_name(data, off) + 4
    ips = []
    for _ in range(ancount):
        off = _skip_dns_name(data, off)
        if off + 10 > len(data):
            break
        rtype, rclass, _ttl, rdlen = struct.unpack("!HHIH", data[off:off + 10])
        off += 10
        rdata = data[off:off + rdlen]
        off += rdlen
        if rtype == 1 and rclass == 1 and rdlen == 4:
            ips.append(socket.inet_ntoa(rdata))
    return ips


def _fallback_resolve_host(host: str) -> list[str]:
    override = os.environ.get("FE_LLM_RESOLVE_IP", "").strip()
    if override:
        return [override]
    if not _dns_fallback_enabled():
        return []
    if host in _DNS_CACHE:
        return list(_DNS_CACHE[host])
    for server in _dns_servers():
        try:
            ips = _query_dns_a(host, server)
        except OSError:
            continue
        if ips:
            _DNS_CACHE[host] = ips
            return list(ips)
    return []


def resolve_llm_host(host: str, port: int = 443) -> tuple[list[str], str]:
    """Resolve an LLM host, falling back to China public DNS when system DNS fails."""
    try:
        infos = socket.getaddrinfo(host, port)
        ips = sorted({info[4][0] for info in infos})
        if ips:
            return ips, "system"
    except socket.gaierror:
        pass
    ips = _fallback_resolve_host(host)
    if ips:
        return ips, "fallback"
    return [], "failed"


@contextmanager
def _patched_getaddrinfo(host: str, ip: str):
    original = socket.getaddrinfo

    def patched(query_host, port, family=0, socktype=0, proto=0, flags=0):
        if query_host == host:
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))]
        return original(query_host, port, family, socktype, proto, flags)

    socket.getaddrinfo = patched
    try:
        yield
    finally:
        socket.getaddrinfo = original


def _uses_socks_proxy(proxy: str) -> bool:
    return urlparse(proxy).scheme.lower().startswith("socks")


def _temperature(model: str | None = None) -> float:
    try:
        return float(os.environ.get("FE_LLM_TEMPERATURE", "0.7"))
    except ValueError:
        return 0.7


def _moonshot_required_temperature(error_text: str) -> float | None:
    m = re.search(r"invalid temperature:\s*only\s*([0-9]+(?:\.[0-9]+)?)\s+is allowed",
                  error_text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _timeout_s() -> int:
    try:
        return int(os.environ.get("FE_LLM_TIMEOUT", "45"))
    except ValueError:
        return 45


def _kimi_thinking() -> str:
    val = os.environ.get("FE_LLM_THINKING", "").strip().lower()
    return val if val in {"enabled", "disabled"} else ""


def _extract_message_content(msg) -> str:
    content = msg.get("content") if isinstance(msg, dict) else ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        content = "".join(parts)
    return str(content or msg.get("reasoning_content") or "")


def _request_json(method: str, url: str, headers: dict, payload: dict | None,
                  timeout: int, proxy: str) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    if proxy and _uses_socks_proxy(proxy):
        try:
            import requests
        except ImportError as e:  # pragma: no cover - depends on optional env
            raise RuntimeError(
                "SOCKS proxy requested but requests[socks] is not installed. "
                "Install requirements.txt or use an HTTP proxy.") from e
        resp = requests.request(method, url, headers=headers, data=data, timeout=timeout,
                                proxies={"http": proxy, "https": proxy})
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        return resp.json()

    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy}) if proxy
        else urllib.request.ProxyHandler({}))
    parsed = urlparse(url)
    patch_ctx = nullcontext()
    if not proxy and parsed.hostname:
        ips, source = resolve_llm_host(parsed.hostname, parsed.port or 443)
        if source == "fallback" and ips:
            patch_ctx = _patched_getaddrinfo(parsed.hostname, ips[0])
    try:
        with patch_ctx:
            with opener.open(req, timeout=timeout) as r:
                return json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {e.code}: {detail}") from e


def _openai_compatible_completion(base: str, system: str, user: str,
                                  model: str, max_tokens: int) -> str:
    key = _resolve_api_key()
    if not key:
        raise RuntimeError("missing FE_LLM_API_KEY / MOONSHOT_API_KEY / KIMI_API_KEY")
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": _temperature(model),
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }
    thinking = _kimi_thinking()
    if thinking and "moonshot" in urlparse(base).netloc:
        body["thinking"] = {"type": thinking}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        data = _request_json("POST", base.rstrip("/") + "/chat/completions", headers,
                             body, _timeout_s(), _proxy_url())
    except RuntimeError as e:
        required_temp = _moonshot_required_temperature(str(e))
        if required_temp is None:
            raise
        body = dict(body)
        body["temperature"] = required_temp
        data = _request_json("POST", base.rstrip("/") + "/chat/completions", headers,
                             body, _timeout_s(), _proxy_url())
    msg = data["choices"][0]["message"]
    global _LAST_LLM_INFO
    _LAST_LLM_INFO = {"provider": _llm_provider() or "openai-compatible",
                      "base_url": base.rstrip("/"), "model": model}
    return _extract_message_content(msg)


def describe_llm_config() -> dict:
    """Serializable, key-safe description of the configured live LLM path."""
    bases = _configured_openai_bases()
    model = _resolve_model(anthropic_provider=not bases)
    return {
        "provider": _llm_provider() or ("openai-compatible" if bases else "anthropic"),
        "base_urls": bases,
        "model": model,
        "api_key_present": bool(_resolve_api_key()) if bases else False,
        "anthropic_key_present": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "proxy": _proxy_url() or "",
        "uses_system_proxy": os.environ.get("FE_LLM_USE_SYSTEM_PROXY") in ("1", "true", "True"),
        "timeout_s": _timeout_s(),
        "temperature": _temperature(model),
        "thinking": _kimi_thinking(),
        "require_llm": os.environ.get("FE_REQUIRE_LLM", "") in ("1", "true", "True"),
        "dns_fallback": _dns_fallback_enabled(),
        "dns_servers": _dns_servers(),
        "last_success": dict(_LAST_LLM_INFO),
    }


def _is_transient_llm_error(errtext: str) -> bool:
    """True for routing/network errors worth retrying; False for auth/model 4xx."""
    t = errtext.lower()
    if "http 4" in t:                      # 401 auth / 404 model / 400 bad-request: don't retry
        return False
    return any(w in t for w in (
        "timed out", "timeout", "handshake", "nodename", "name or service",
        "connection refused", "connection reset", "reset by peer", "unreachable",
        "temporarily", "broken pipe", "remote end closed", "eof occurred",
        "http 5", "http 429", "bad gateway", "service unavailable", "gateway time"))


def _call_llm(system: str, user: str, model=None, max_tokens=4000) -> str | None:
    """Call the configured LLM and return text.

    Provider resolution:
      1. OpenAI-compatible bases from FE_LLM_BASE_URLS / FE_LLM_BASE_URL /
         MOONSHOT_BASE_URL / OPENAI_BASE_URL.  For Kimi China access, use
         FE_LLM_PROVIDER=kimi-cn or FE_LLM_BASE_URL=https://api.moonshot.cn/v1.
      2. Anthropic SDK when no OpenAI-compatible base is configured and
         ANTHROPIC_API_KEY is set.
      3. None, so the engine can use the deterministic macro library.
    """
    bases = _configured_openai_bases()
    if bases:
        model = _resolve_model(model)
        retries = max(0, int(os.environ.get("FE_LLM_RETRIES", "4")))
        last_errors = []
        for attempt in range(retries + 1):
            errors = []
            for base in bases:
                try:
                    return _openai_compatible_completion(base, system, user, model, max_tokens)
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{base}: {type(e).__name__}: {str(e)[:240]}")
            last_errors = errors
            joined = " | ".join(errors)
            # retry only transient routing/network errors (not auth/model 4xx)
            if attempt < retries and _is_transient_llm_error(joined):
                back = min(2.0 * (2 ** attempt), 20.0)
                print(f"  [LLM transient failure (attempt {attempt+1}/{retries+1}); "
                      f"retrying in {back:.0f}s] {joined[:120]}", flush=True)
                time.sleep(back)
                continue
            break
        raise RuntimeError("OpenAI-compatible LLM calls failed; tried "
                           + " | ".join(last_errors))
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        model = _resolve_model(model, anthropic_provider=True)
        client = anthropic.Anthropic()
        resp = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                      messages=[{"role": "user", "content": user}])
        global _LAST_LLM_INFO
        _LAST_LLM_INFO = {"provider": "anthropic", "base_url": "", "model": model}
        return "".join(getattr(b, "text", "") for b in resp.content)
    return None


def llm_mutation(node, root_code, coe_context, metrics_desc, model=None,
                 max_tokens=4000) -> Mutation | None:
    """Ask the configured LLM for a mutation; None on failure (engine falls back)."""
    user = f"""# Original Program
```python
{root_code}
```

# Current Program (evolve THIS one)
```python
{node.code}
```

# Current Program Information
- Metrics: {metrics_desc}
- Fitness: {getattr(node, 'fitness', float('nan')):+.4f}

# Program Evolution History (chain of experience)
{coe_context}

# Task
Suggest targeted improvements to the CURRENT program to raise its metrics and fitness.
{RESPONSE_FORMAT}
"""
    try:
        text = _call_llm(SYSTEM_PROMPT, user, model, max_tokens)
    except Exception as e:  # noqa: BLE001
        return Mutation(node.code, {}, f"(LLM error: {type(e).__name__}: {str(e)[:220]})", "", "llm_error")
    if not text:
        return None

    new_code = _parse_search_replace(text, node.code)
    parse_mode = "diff"
    if not new_code:
        new_code = _parse_full_code(text)
        parse_mode = "full_code"
    if not new_code or new_code == node.code:
        _dump_llm_failure(text, "no applicable SEARCH/REPLACE diff or full code")
        return None
    try:
        compile_factor(new_code)
    except FactorRunError as e:
        _dump_llm_failure(text, f"compiled mutation failed: {e}")
        return None
    pspace = _parse_params(text)
    pspace = _auto_declare_params(new_code, pspace)   # V3 #6: no hidden knobs
    return Mutation(new_code, pspace,
                    _section(text, "IDEA") or _section(text, "Analyse"),
                    f"LLM {parse_mode}", "llm")
