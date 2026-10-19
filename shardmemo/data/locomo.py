"""LoCoMo (Maharana et al., 2024): Tier-B evidence in three shard families (profile facts,
observation turns, session summaries), Tier-A session anchors and scoped QA requests."""
from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..memory import EvidenceItem
from ..scope import Metadata, Request
from ..tier_a import ActiveContextStore

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
CATEGORY = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}
_DIA = re.compile(r"D(\d+):(\d+)")


def parse_session_time(s: str) -> datetime | None:
    """'1:56 pm on 8 May, 2023' -> datetime."""
    s = s.strip()
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %B %Y", "%d %B, %Y", "%d %B %Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    m = re.search(r"(\d{1,2}) ([A-Za-z]+),? (\d{4})", s)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%d %B %Y")
        except ValueError:
            return None
    return None


def load_locomo(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LOCOMO_URL, p)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class LocomoData:
    items: list[EvidenceItem]
    requests: list[Request]
    anchors: dict[str, dict[int, datetime]]
    speakers: dict[str, tuple[str, str]]
    convs: list[dict] = field(default_factory=list)

    def split(self, test_convs: Sequence[str]) -> tuple[list[Request], list[Request]]:
        test = set(test_convs)
        return [r for r in self.requests if r.conv_id not in test], [r for r in self.requests if r.conv_id in test]


def _sessions(conv: dict) -> list[int]:
    return sorted(int(k.split("_")[1]) for k in conv if re.fullmatch(r"session_\d+", k))


def _turn_text(turn: dict) -> str:
    text = turn["text"]
    if turn.get("blip_caption"):
        text = f"{text} [shared image: {turn['blip_caption']}]"
    return f"{turn['speaker']}: {text}"


def build_items(convs: Iterable[dict], session_only: bool = False, show_previous_turn: bool = False
                ) -> tuple[list[EvidenceItem], dict[str, dict[int, datetime]], dict[str, tuple[str, str]]]:
    """Tier-B evidence items. session_only=True puts every turn in one session family and drops the
    profile and summary families. show_previous_turn appends the preceding turn to the text shown
    to the generator (the indexed text is unchanged)."""
    items: list[EvidenceItem] = []
    anchors: dict[str, dict[int, datetime]] = {}
    speakers: dict[str, tuple[str, str]] = {}
    for sample in convs:
        cid = sample["sample_id"]
        conv = sample["conversation"]
        speakers[cid] = (conv["speaker_a"], conv["speaker_b"])
        anchors[cid] = {}
        for sid in _sessions(conv):
            date = conv.get(f"session_{sid}_date_time")
            when = parse_session_time(date) if date else None
            if when:
                anchors[cid][sid] = when
            turns = conv[f"session_{sid}"]
            for ti, turn in enumerate(turns):
                text = _turn_text(turn)
                shown = text
                if show_previous_turn and ti > 0:
                    shown = f"{text} (in reply to {_turn_text(turns[ti - 1])})"
                items.append(EvidenceItem(
                    uid=f"{cid}:{turn['dia_id']}", text=text, display=f"(session {sid}) {shown}",
                    meta=Metadata.make(conv_id=cid, speaker=turn["speaker"], session_id=sid,
                                       family="session" if session_only else "obs", t=when),
                    source_ids=(f"{cid}:{turn['dia_id']}",), time=when, unit_key=(cid, sid)))
        if session_only:
            continue
        items += _profile_items(sample, cid, anchors[cid])
        items += _summary_items(sample, cid, anchors[cid], speakers[cid])
    return items, anchors, speakers


def _profile_items(sample: dict, cid: str, anchors: dict[int, datetime]) -> list[EvidenceItem]:
    items = []
    for key, per_speaker in sample.get("observation", {}).items():
        m = re.search(r"session_(\d+)", key)
        sid = int(m.group(1)) if m else 0
        when = anchors.get(sid)
        for sp, facts in per_speaker.items():
            for i, fact in enumerate(facts):
                text, ev = (fact[0], fact[1]) if isinstance(fact, (list, tuple)) and len(fact) >= 2 else (str(fact), "")
                src = tuple(f"{cid}:D{a}:{b}" for a, b in _DIA.findall(str(ev)))
                items.append(EvidenceItem(
                    uid=f"{cid}:profile:{sid}:{sp}:{i}", text=text,
                    display=f"(session {sid}) fact about {sp}: {text}",
                    meta=Metadata.make(conv_id=cid, speaker=sp, session_id=sid, family="profile", t=when),
                    source_ids=src, time=when, unit_key=(cid, sp)))
    return items


def _summary_items(sample: dict, cid: str, anchors: dict[int, datetime],
                   speakers: tuple[str, str]) -> list[EvidenceItem]:
    """Session summaries split into chunks of three sentences."""
    items = []
    for key, summary in sample.get("session_summary", {}).items():
        m = re.search(r"session_(\d+)", key)
        sid = int(m.group(1)) if m else 0
        when = anchors.get(sid)
        sents = re.split(r"(?<=[.!?])\s+", str(summary).strip())
        for i in range(0, len(sents), 3):
            chunk = " ".join(sents[i:i + 3]).strip()
            if not chunk:
                continue
            items.append(EvidenceItem(
                uid=f"{cid}:session:{sid}:{i // 3}", text=chunk,
                display=f"(session {sid}) session summary: {chunk}",
                meta=Metadata(conv_ids=frozenset([cid]), speakers=frozenset(speakers),
                              session_ids=frozenset([sid]), family="session", t_min=when, t_max=when),
                source_ids=(), time=when, unit_key=(cid, sid)))
    return items


def referenced_speakers(question: str, names: Sequence[str]) -> frozenset:
    return frozenset(n for n in names if re.search(rf"\b{re.escape(n)}\b", question, re.I))


def build_requests(convs: Iterable[dict], speakers: dict[str, tuple[str, str]]) -> list[Request]:
    """QA requests without the adversarial category."""
    reqs: list[Request] = []
    for sample in convs:
        cid = sample["sample_id"]
        for qi, qa in enumerate(sample["qa"]):
            cat = int(qa.get("category", 0))
            if cat == 5:
                continue
            ans = qa.get("answer", qa.get("adversarial_answer", ""))
            ev = [f"{cid}:D{a}:{b}" for e in qa.get("evidence", []) for a, b in _DIA.findall(str(e))]
            reqs.append(Request(
                text=qa["question"], conv_id=cid,
                speakers_referenced=referenced_speakers(qa["question"], speakers[cid]),
                meta={"qid": f"{cid}:q{qi}", "answer": str(ans), "evidence": ev, "category": cat,
                      "category_name": CATEGORY.get(cat, str(cat))}))
    return reqs


def build_active_context(store: ActiveContextStore, anchors: dict[str, dict[int, datetime]]) -> None:
    """Tier A: one date anchor per session."""
    for cid, sessions in anchors.items():
        for sid, when in sessions.items():
            store.add_session_anchor(cid, sid, when)


def prepare_locomo(path: str | Path, session_only: bool = False, show_previous_turn: bool = False) -> LocomoData:
    convs = load_locomo(path)
    items, anchors, speakers = build_items(convs, session_only=session_only, show_previous_turn=show_previous_turn)
    return LocomoData(items=items, requests=build_requests(convs, speakers), anchors=anchors, speakers=speakers,
                      convs=convs)
