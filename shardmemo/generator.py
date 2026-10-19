"""Answer generation through an OpenAI-compatible chat API with a prompt-hash response cache."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from .config import GeneratorConfig
from .metrics import normalize_answer

PROMPTS = {
    "locomo": """You answer questions about a long-term conversation between two people, using retrieved memories.
Each memory starts with its session number and, when known, the session date.

Rules for the final answer:
- A short phrase (usually 1-6 words), never a full sentence.
- Dates: give absolute dates like "7 May 2023". Convert relative expressions using the memory's session date:
  "yesterday" -> the day before; "last Sunday" -> "the Sunday before 8 May 2023"; "last week" -> "the week before 8 May 2023";
  "last year" -> the previous year (e.g. 2022); "two years ago" -> that year.
- If the answer has several items, list them separated by commas.
- If the question asks what someone would likely think, prefer or do, infer it from the memories.
- If the memories do not state the answer, give your best short guess based on them.

Background:
{context}

Memories (most relevant first):
{evidence}

Question: {question}

First write one line starting with "Reasoning:" that names the relevant memories (and does any date arithmetic).
Then write one line starting with "Answer:" containing only the short answer.""",
    "hotpotqa": """Answer the multi-hop question using the retrieved evidence sentences (each starts with its document title).

Rules for the final answer:
- Copy the answer span exactly as it is written in the evidence whenever possible; keep it as short as possible (an entity, name, number, date or short phrase), never a sentence, and add no qualifiers the question does not ask for.
- For yes/no questions (e.g. "Are both ...", "Is X ...", "Did ...") answer exactly "yes" or "no".
- For comparisons ("which of the two", "who was born first", "which is older") give only the chosen name.
- If the evidence is incomplete, combine it with what you know and give your best guess; never answer "unknown".

Evidence:
{evidence}

Question: {question}

First write one line starting with "Reasoning:" that names the bridge entity or the compared facts.
Then write one line starting with "Answer:" containing only the answer span.""",
}


def extract_answer(text: str) -> str:
    """Text after the last 'Answer:' marker, otherwise the first line."""
    t = (text or "").strip()
    markers = list(re.finditer(r"(?im)^\s*(?:final\s+)?(?:short\s+)?answer\s*:\s*", t))
    if markers:
        t = t[markers[-1].end():]
    return t.split("\n")[0].strip().strip('"').strip()


class Generator:
    def __init__(self, cfg: GeneratorConfig, cache_path: str | None = None, max_retries: int = 6):
        from openai import OpenAI
        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(f"environment variable {cfg.api_key_env} is not set")
        if not cfg.model:
            raise RuntimeError("environment variable LLM_MODEL is not set")
        self.name = cfg.model
        self.model = cfg.model
        self.temperature = 0.0
        self.max_tokens = cfg.max_tokens
        self.n_samples = cfg.n_samples
        self.sample_temperature = cfg.sample_temperature
        self.workers = cfg.workers
        self.max_retries = max_retries
        self._client = OpenAI(base_url=cfg.base_url or None, api_key=api_key, timeout=120)
        self._lock = threading.Lock()
        self.cache_path = cache_path
        self._cache = self._load_cache(cache_path)
        self.n_calls = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    @staticmethod
    def _load_cache(path: str | None) -> dict:
        cache = {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        cache[rec["k"]] = rec["v"]
                    except (json.JSONDecodeError, KeyError):
                        continue
        return cache

    @staticmethod
    def _message(question: str, evidence: Sequence[str], context: str, prompt: str) -> str:
        ev = "\n".join(f"{i + 1}. {e}" for i, e in enumerate(evidence)) or "(none)"
        return prompt.format(context=context or "(none)", evidence=ev, question=question)

    def _key(self, msg: str) -> str:
        return hashlib.sha1(f"{self.model}|{self.temperature}|{self.max_tokens}|{msg}".encode()).hexdigest()

    def _call(self, msg: str, temperature: float | None = None) -> str:
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model, temperature=self.temperature if temperature is None else temperature,
                    max_tokens=self.max_tokens, messages=[{"role": "user", "content": msg}])
                with self._lock:
                    self.n_calls += 1
                    if resp.usage is not None:
                        self.usage["prompt_tokens"] += resp.usage.prompt_tokens or 0
                        self.usage["completion_tokens"] += resp.usage.completion_tokens or 0
                return extract_answer(resp.choices[0].message.content or "")
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60)
        return ""

    def _store(self, key: str, out: str) -> None:
        with self._lock:
            self._cache[key] = out
            if self.cache_path:
                os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": key, "v": out}) + "\n")

    def _fill(self, todo: dict, temperature: float | None = None) -> None:
        if not todo:
            return
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for key, out in zip(todo, ex.map(lambda m: self._call(m, temperature), todo.values())):
                self._store(key, out)

    def generate(self, question: str, evidence: Sequence[str], context: str, prompt: str) -> str:
        return self.generate_many([(question, evidence, context, prompt)])[0]

    def generate_many(self, jobs: Sequence[tuple]) -> list:
        """jobs: (question, evidence, context, prompt) tuples -> answers in the same order."""
        msgs = [self._message(*j) for j in jobs]
        keys = [self._key(m) for m in msgs]
        self._fill({k: m for k, m in zip(keys, msgs) if k not in self._cache})
        greedy = [self._cache[k] for k in keys]
        if self.n_samples <= 1:
            return greedy
        sample_keys = {(k, i): f"{k}|s{i}|t{self.sample_temperature}" for k in keys for i in range(self.n_samples)}
        msg_of = dict(zip(keys, msgs))
        todo = {sk: msg_of[k] for (k, _), sk in sample_keys.items() if sk not in self._cache}
        self._fill(todo, self.sample_temperature)
        return [self._vote(g, [self._cache[sample_keys[(k, i)]] for i in range(self.n_samples)])
                for k, g in zip(keys, greedy)]

    @staticmethod
    def _vote(greedy: str, samples: Sequence[str]) -> str:
        """Majority vote over normalised answers; ties keep the greedy answer when it is among the winners."""
        cands = [greedy] + list(samples)
        counts = Counter(normalize_answer(c) for c in cands if c)
        if not counts:
            return greedy
        top = max(counts.values())
        winners = {a for a, c in counts.items() if c == top}
        if normalize_answer(greedy) in winners:
            return greedy
        return next(c for c in cands if normalize_answer(c) in winners)
