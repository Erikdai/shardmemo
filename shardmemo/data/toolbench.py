"""ToolBench-style procedural memory benchmark: a deterministic multi-family tool environment with
list -> lookup procedures, noisy trajectories, in-domain and family-level OOD splits, authorised
namespaces per request and API-spec documents for the Tier-B fallback. The environment is simulated
and does not call the real ToolBench (RapidAPI) endpoints; the full environment is available from the
official ToolBench repository, https://github.com/OpenBMB/ToolBench."""
from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..memory import EvidenceItem
from ..scope import Metadata, Request
from ..tier_c import ToolCall, Trace


def _h(s: str, mod: int) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod


# Each family: entity pool, query templates, API description and its list -> lookup procedure.

FAMILY_SPECS: dict[str, dict] = {
    "transitaires": dict(
        entities=["ACT Agence Caledonienne de Transit", "Pacific Freight Services", "Noumea Customs Brokers",
                  "Caledonia Transit Express", "Lagoon Logistics", "Koumac Shipping Agency", "Transit Sud Pacifique",
                  "Bourail Douane Services", "Mont-Dore Freight", "Tontouta Air Cargo"],
        templates=["Get contact details of the {e} customs agency in New Caledonia.",
                   "I need the phone and email of {e}, a customs broker in New Caledonia.",
                   "Find the postal code and email for the freight forwarder {e} in New Caledonia."],
        spec="transitaires API: lists customs agencies (transitaires) and returns contact details by id. "
             "Covers customs agencies in New Caledonia (country code NC) only.",
        list_tool="list_transitaires", lookup_tool="transitaire_for_transitaires", key="is_id", list_arg=None,
    ),
    "weather": dict(
        entities=["Paris", "Sydney", "Tokyo", "Nairobi", "Lima", "Toronto", "Oslo", "Cairo", "Mumbai", "Santiago"],
        templates=["What is the current weather in {e}?", "Get the forecast for {e} today.",
                   "Check the temperature and conditions in {e}."],
        spec="weather API: geocodes a city name to coordinates and returns the current forecast. "
             "Covers major cities worldwide; historical data is not available.",
        list_tool="geocode", lookup_tool="forecast", key=("lat", "lon"), list_arg="city",
    ),
    "finance": dict(
        entities=["Acme Robotics", "Blue Harbor Energy", "Cobalt Foods", "Delta Quantum", "Evergreen Mining",
                  "Fjord Shipping", "Granite Bank", "Helios Solar", "Iris Biotech", "Juniper Retail"],
        templates=["Get the latest stock quote for {e}.", "What is {e} trading at right now?",
                   "Look up the share price of {e}."],
        spec="finance API: resolves a company name to a ticker symbol and returns a delayed quote. "
             "Equities only; no crypto or FX.",
        list_tool="symbol_search", lookup_tool="quote", key="symbol", list_arg="company",
    ),
    "movies": dict(
        entities=["Silent Harbor", "The Last Orchard", "Neon Tide", "Glass Meridian", "Paper Comets",
                  "Winter Cartographer", "Hollow Signal", "Amber Protocol", "Salt and Static", "The Ninth Lantern"],
        templates=["Who directed the movie {e} and what is its rating?", "Get details about the film {e}.",
                   "Find the release year and director of {e}."],
        spec="movies API: searches titles and returns details (year, director, rating) by movie id. "
             "Feature films only; TV series are not covered.",
        list_tool="search_title", lookup_tool="movie_details", key="movie_id", list_arg="title",
    ),
    "geo": dict(
        entities=["Portugal", "Kenya", "Chile", "Vietnam", "Norway", "Morocco", "Peru", "Thailand", "Iceland", "Ghana"],
        templates=["What currency does {e} use and what is its USD rate?",
                   "Get the currency and exchange rate for {e}.",
                   "Find the capital and currency rate of {e}."],
        spec="geo API: returns country facts (capital, ISO code, currency) and a USD exchange rate per currency code. "
             "Sovereign countries only; overseas territories are not covered.",
        list_tool="country_info", lookup_tool="currency_rate", key="currency_code", list_arg="country",
    ),
    "books": dict(
        entities=["Tidewater Letters", "The Cartography of Rain", "Small Engines", "A Field Guide to Ghosts",
                  "Northern Arithmetic", "The Orchard Keeper's Son", "Blue Hours", "Iron Lullaby",
                  "Maps for Lost Cities",
                  "The Quiet Reactor"],
        templates=["Who wrote the book {e} and how many pages does it have?", "Get details for the novel {e}.",
                   "Find the author and year of {e}."],
        spec="books API: searches book titles and returns bibliographic details by ISBN. "
             "English-language editions only.",
        list_tool="search_book", lookup_tool="book_details", key="isbn", list_arg="title",
    ),
    # family-level OOD
    "flights": dict(
        entities=["Lisbon", "Auckland", "Denver", "Jakarta", "Vienna", "Accra", "Bogota", "Seoul", "Dublin", "Perth"],
        templates=["Get the IATA code and departure status for flights from {e} today.",
                   "Check today's departures from {e} airport.", "Look up the airport code for {e} and its delays."],
        spec="flights API: resolves a city to its main airport code and returns departure status. "
             "Commercial airports only.",
        list_tool="airport_code", lookup_tool="departures", key="iata", list_arg="city",
    ),
    "recipes": dict(
        entities=["Shakshuka", "Pad Thai", "Ratatouille", "Bibimbap", "Jollof Rice", "Ceviche", "Moussaka",
                  "Pho", "Feijoada", "Tagine"],
        templates=["Get the ingredients and cooking time for {e}.", "Find a recipe for {e} with its calories.",
                   "Look up how long it takes to cook {e}."],
        spec="recipes API: searches dishes and returns ingredients, cooking time and calories by recipe id. "
             "Savoury dishes only.",
        list_tool="search_recipe", lookup_tool="recipe_details", key="recipe_id", list_arg="dish",
    ),
}
IN_DOMAIN = ["transitaires", "weather", "finance", "movies", "geo", "books"]
OOD = ["flights", "recipes"]

# Providers of the same function; each provider is a separate tool namespace with its own endpoint
# names, ids and payloads.
PROVIDERS: dict[str, list[str]] = {
    "transitaires": [""],
    "weather": ["skycast", "meteobox", "airgauge", "cloudline"],
    "finance": ["quotely", "tickerhub", "marketline", "stockpulse"],
    "movies": ["filmdex", "cinebase", "reelinfo", "screenly"],
    "geo": ["atlasly", "countrykit", "geofacts", "mapmeta"],
    "books": ["shelfapi", "bookbase", "pagefinder", "inkdex"],
    "flights": ["flightboard", "aerostat", "gatecheck", "skyroute"],
    "recipes": ["cookdex", "dishbase", "kitchenapi", "platebook"],
}


def namespace(fam: str, provider: str) -> str:
    return fam if not provider else f"{fam}.{provider}"


def all_namespaces(fams: Sequence[str]) -> list[tuple[str, str]]:
    return [(f, p) for f in fams for p in PROVIDERS[f]]


class SimulatedToolEnv:
    """Deterministic tool server; `call(tool, args)` returns {"error": ...} on bad input. The first
    step of every provider has three interchangeable aliases, so several procedures solve a task."""

    ALIASES = ("", "_v2", "_by_name")

    def __init__(self, single_provider: bool = False):
        self.single_provider = single_provider
        self.tools: dict[str, tuple[str, str, str]] = {}      # tool name -> (family, provider, base tool)
        for fam, spec in FAMILY_SPECS.items():
            for prov in self.providers(fam):
                pre = f"{prov}_" if prov else ""
                for a in self.ALIASES:
                    self.tools[pre + spec["list_tool"] + a] = (fam, prov, spec["list_tool"])
                self.tools[pre + spec["lookup_tool"]] = (fam, prov, spec["lookup_tool"])
        self.calls = 0

    def providers(self, fam: str) -> list[str]:
        return [""] if self.single_provider else PROVIDERS[fam]

    @staticmethod
    def tool_name(prov: str, base: str) -> str:
        return f"{prov}_{base}" if prov else base

    def entity_id(self, fam: str, e: str, prov: str = "") -> str:
        if fam == "transitaires":
            return e.upper().replace(" ", "_")
        return f"{(prov or fam)[:3]}_{_h(prov + fam + e, 10 ** 6):06d}"

    def call(self, tool: str, args: dict[str, Any]) -> Any:
        self.calls += 1
        hit = self.tools.get(tool)
        if hit is None:
            return {"error": f"unknown tool {tool}"}
        fam, prov, tool = hit
        spec = FAMILY_SPECS[fam]
        ents = spec["entities"]
        salt = prov
        if tool == spec["list_tool"]:
            if fam == "transitaires":
                if str(args.get("country", "NC")).upper() != "NC":
                    return {"error": "country not covered"}
                return [{"id": self.entity_id(fam, e, prov), "name": e, "country": "NC"} for e in ents]
            name = str(args.get(spec["list_arg"], ""))
            match = next((e for e in ents if e.lower() == name.lower()), None)
            if match is None:
                match = next((e for e in ents
                              if name and (name.lower() in e.lower() or e.lower() in name.lower())), None)
            if match is None:
                return {"error": f"{spec['list_arg']} not found: {name}"}
            if fam == "weather":
                return {"city": match, "lat": round(-60 + _h(salt + match, 12000) / 100, 2),
                        "lon": round(-180 + _h(salt + match + "x", 36000) / 100, 2)}
            key = spec["key"]
            return {key: self.entity_id(fam, match, prov), "name": match}
        # lookup tool
        if fam == "weather":
            if args.get("lat") is None or args.get("lon") is None:
                return {"error": "lat/lon required"}
            k = f"{salt}:{args['lat']}:{args['lon']}"
            return {"temp_c": _h(k, 40) - 5, "condition": ["clear", "cloudy", "rain", "windy"][_h(k + "c", 4)]}
        key = spec["key"]
        ident = args.get(key)
        ent = next((e for e in ents if self.entity_id(fam, e, prov) == ident), None)
        if ent is None:
            return {"error": f"unknown {key}: {ident}"}
        e2 = salt + ent
        if fam == "transitaires":
            return {"codePostal": str(98800 + _h(ent, 90)), "email": f"sales@{ent.split()[0].lower()}.nc",
                    "phone": f"+687 {_h(ent, 90) + 10}.{_h(ent + 'p', 90) + 10}.{_h(ent + 'q', 90) + 10}"}
        if fam == "finance":
            return {"symbol": ident, "price": round(10 + _h(e2, 50000) / 100, 2), "currency": "USD"}
        if fam == "movies":
            return {"title": ent, "year": 1980 + _h(ent, 45), "rating": round(5 + _h(e2 + "r", 50) / 10, 1),
                    "director": ["A. Moreau", "K. Tanaka", "L. Okafor", "S. Petrov"][_h(ent + "d", 4)], "source": prov}
        if fam == "geo":
            return {"code": ident, "rate_usd": round(0.01 + _h(e2, 2000) / 100, 2)}
        if fam == "books":
            return {"title": ent, "author": ["M. Alvarez", "R. Chen", "D. Osei", "H. Lindqvist"][_h(ent, 4)],
                    "year": 1950 + _h(ent + "y", 74), "pages": 120 + _h(ent + "p", 700), "source": prov}
        if fam == "flights":
            return {"iata": ident, "delayed": _h(e2, 30), "on_time": 100 + _h(e2 + "o", 300)}
        if fam == "recipes":
            return {"dish": ent, "minutes": 15 + _h(e2, 120), "calories": 200 + _h(e2 + "k", 700),
                    "ingredients": ["salt", "oil", ["onion", "garlic", "tomato", "rice"][_h(ent, 4)]]}
        return {"error": "unreachable"}

    def gold_trace(self, fam: str, entity: str, query: str, task_id: str, rng: random.Random, prov: str = "") -> Trace:
        spec = FAMILY_SPECS[fam]
        calls: list[ToolCall] = []
        outs: list[Any] = []
        errs: list[bool] = []

        def add(call: ToolCall, out: Any, failed: bool) -> None:
            calls.append(call)
            outs.append(out)
            errs.append(failed)

        list_tool = self.tool_name(prov, spec["list_tool"] + rng.choice(self.ALIASES))
        lookup_tool = self.tool_name(prov, spec["lookup_tool"])
        # failed and redundant calls, as in recorded tool-use trajectories
        if rng.random() < 0.6:
            bad = ToolCall(lookup_tool, {spec["key"] if isinstance(spec["key"], str) else "lat": "unknown"})
            add(bad, self.call(bad.tool, bad.args), True)
        if rng.random() < 0.3:
            ofam = rng.choice([f for f in FAMILY_SPECS if f != fam])
            other = FAMILY_SPECS[ofam]
            bad = ToolCall(self.tool_name(self.providers(ofam)[0], other["list_tool"]),
                           {other["list_arg"] or "country": entity})
            add(bad, self.call(bad.tool, bad.args), True)
        if fam == "transitaires":
            c1 = ToolCall(list_tool, {"country": "NC"})
        else:
            c1 = ToolCall(list_tool, {spec["list_arg"]: entity})
        o1 = self.call(c1.tool, c1.args)
        add(c1, o1, False)
        if rng.random() < 0.4:
            add(ToolCall(c1.tool, dict(c1.args)), o1, False)
        if fam == "weather":
            c2 = ToolCall(lookup_tool, {"lat": o1["lat"], "lon": o1["lon"]})
        elif fam == "transitaires":
            c2 = ToolCall(lookup_tool, {"is_id": self.entity_id(fam, entity, prov)})
        else:
            c2 = ToolCall(lookup_tool, {spec["key"]: o1[spec["key"]]})
        o2 = self.call(c2.tool, c2.args)
        add(c2, o2, False)
        if rng.random() < 0.4:
            add(ToolCall(c2.tool, dict(c2.args)), o2, False)
        return Trace(task_id=task_id, query=query, calls=calls, outputs=outs, final=o2,
                     meta=Metadata.make(namespace=namespace(fam, prov), family="skill"), errors=errs)


@dataclass
class ToolTask:
    task_id: str
    family: str
    entity: str
    query: str
    trace: Trace
    gold: Any
    n_gold_steps: int
    allowed_namespaces: frozenset
    gold_in_scope: bool
    provider: str = ""

    @property
    def namespace(self) -> str:
        return namespace(self.family, self.provider)

    def request(self) -> Request:
        return Request(text=self.query, allowed_namespaces=self.allowed_namespaces, tier_hint="C",
                       meta={"task_id": self.task_id, "family": self.family, "gold": self.gold,
                             "n_gold_steps": self.n_gold_steps, "gold_in_scope": self.gold_in_scope,
                             "namespace": self.namespace})


@dataclass
class ToolBenchData:
    env: SimulatedToolEnv
    train: list[ToolTask]
    test: list[ToolTask]
    ood: list[ToolTask]
    spec_items: list[EvidenceItem] = field(default_factory=list)


def make_toolbench(seed: int = 0, n_in_domain: int = 348, n_ood: int = 181, train_frac: float = 0.7,
                   p_unauthorized: float = 0.09, single_provider: bool = False) -> ToolBenchData:
    """Induction / evaluation split over the in-domain families plus a family-level OOD split.
    A request authorises the provider namespace its task needs and one unrelated namespace; with
    probability p_unauthorized only the unrelated one, so the gold skill is out of scope.
    single_provider=True uses one namespace per family."""
    rng = random.Random(seed)
    env = SimulatedToolEnv(single_provider=single_provider)
    every_ns = [namespace(f, p) for f in FAMILY_SPECS for p in env.providers(f)]

    def _tasks(fams: Sequence[str], n: int, prefix: str) -> list[ToolTask]:
        out = []
        for i in range(n):
            fam = fams[i % len(fams)]
            prov = rng.choice(env.providers(fam))
            spec = FAMILY_SPECS[fam]
            ent = rng.choice(spec["entities"])
            query = rng.choice(spec["templates"]).format(e=ent)
            tid = f"{prefix}_{i}"
            tr = env.gold_trace(fam, ent, query, tid, rng, prov=prov)
            other = rng.choice([ns for ns in every_ns if ns.split(".")[0] != fam])
            unauthorized = rng.random() < p_unauthorized
            allowed = frozenset([other] if unauthorized else [namespace(fam, prov), other])
            out.append(ToolTask(tid, fam, ent, query, tr, tr.final, tr.n_steps, allowed, not unauthorized, prov))
        return out

    ind = _tasks(IN_DOMAIN, n_in_domain, "id")
    rng.shuffle(ind)
    n_train = int(round(train_frac * len(ind)))
    train, test = ind[:n_train], ind[n_train:]
    for t in train:
        t.allowed_namespaces = frozenset([t.namespace]) | t.allowed_namespaces
        t.gold_in_scope = True
    ood = _tasks(OOD, n_ood, "ood")
    spec_items = []
    for fam, spec in FAMILY_SPECS.items():
        for prov in env.providers(fam):
            ns = namespace(fam, prov)
            txt = spec["spec"] if not prov else f"{prov} {spec['spec']}"
            spec_items.append(EvidenceItem(uid=f"spec:{ns}", text=txt, display=f"[doc:{ns}/spec] {txt}",
                                           meta=Metadata.make(namespace=ns, family="spec"), source_ids=(ns,),
                                           unit_key=(ns,)))
    return ToolBenchData(env=env, train=train, test=test, ood=ood, spec_items=spec_items)


def skill_valid_for(tier_c, skill, task: ToolTask, filler=None) -> bool:
    """True if executing the skill with slots filled from the task query reproduces the gold answer."""
    slots, ok = (filler or tier_c.filler).fill(task.request(), skill)
    if not ok or not tier_c.validate_preconditions(skill, slots):
        return False
    done, y, _ = tier_c.execute(skill.procedure, slots)
    return bool(done) and y == task.gold
