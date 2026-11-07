"""Model-family assignment for family-holdout experiments.

Three strategies; family_scheme(dataset) picks one by dataset name:
  - keyword: match base-architecture keywords (and known fine-tune
    lineages) in the lowercased model name.  Used for MMLU, Merged and
    BenchPress, where names are heterogeneous fine-tunes but the base
    architecture is almost always present in the name, and for every
    dataset without a rule below, e.g. the Open LLM Leaderboard matrices
    olv1items and olv2sub (HF repo ids from the same leaderboard as MMLU;
    their HF org is the uploader of a fine-tune or merge, not its family:
    the largest olv2sub "orgs" are individual mergers).
  - org: use the HuggingFace org prefix ("org/model").  Used for MTEB and
    the matrices derived from it (any name starting with an ORG_DATASETS
    entry, e.g. mteb2026), where the org identifies the embedding-model
    provider/family.
  - meta_org: the developer of the model's base model, from the
    organization recorded per model in data/<name>.meta.json, field
    META_ORG_FIELDS[prefix] for names starting with prefix: Epoch AI's
    "organization" (epoch) and HELM's "creator_organization" (helm_lite,
    helm_mmlu, helm_classic).  Their model names are display names
    without an org prefix ("Claude 2.1", "Yi (34B)", "o3"), where
    keywords miss whole lineages (OpenAI's o-series, Mistral's
    Magistral; 42% of HELM Classic unmatched), while the curated field
    covers every HELM model and 94% of Epoch's.  Rules, in order
    (meta_org_assignments):
      1. lineage: a fine-tune, distillation or reproduction of another
         developer's base model (LINEAGE_KEYWORDS: Vicuna, Alpaca, Stable
         Beluga, OpenLLaMA -> LLaMA; DeepSeek-R1-Distill-Qwen -> Qwen)
         belongs to the base model's developer, as under the keyword rule
         (which files Vicuna and Alpaca under llama);
      2. organization: a comma-separated list ("DeepSeek,Peking
         University") counts for its first organization; the label is
         its slug (org_family: lowercase, runs of other characters ->
         "-", e.g. "Mistral AI" -> "mistral-ai"), and ORG_ALIASES merges
         spellings of one developer (Epoch: "Google DeepMind" and
         "Google" -> google, "Microsoft Research" -> microsoft);
      3. no organization (17 Epoch models, e.g. PaLM 2, Qwen2.5-Coder):
         the developer of the model's keyword family (keyword_family,
         BASE_DEVELOPERS); "other" if that family has no entry (Epoch:
         RedPajama-INCITE-7B-Base only).
    For rules 1 and 3 the developer label is the first BASE_DEVELOPERS
    candidate that some model of the dataset carries through rule 2
    (Meta is "meta-ai" on Epoch, "meta" on HELM), else the first one.
    A family is thus a developer: its distinct lineages stay together
    (HELM Classic's meta holds OPT and LLaMA, Epoch's nvidia its
    Nemotron models, which the keyword rule files under llama), and a
    fine-tune leaves its fine-tuner's family for its base developer's.

Unmatched models map to "other"; they are never held out but always
remain in the training pool.  The rules of the datasets of the paper
(mmlu, mteb, merged, benchpress) are unchanged.

Run as a script to print family histograms per dataset
(python model_families.py [dataset ...]).
"""

import json
import re

# Ordered: first match wins.  Later entries catch fine-tune lineages
# whose base architecture is known but absent from the name.
KEYWORD_FAMILIES = [
    ("llama", ["llama", "vicuna", "tulu", "athene", "nemotron",
               "alpaca", "wizardlm", "hermes-2-theta", "openchat"]),
    ("mistral", ["mistral", "mixtral", "zephyr", "openhermes",
                 "starling", "snorkel"]),
    ("qwen", ["qwen", "qwq", "sky-t1"]),
    ("gpt", ["gpt-", "gpt3", "gpt4", "chatgpt", "o1-", "o3-", "o4-",
             "davinci", "o1 ", "o3 ", "o4 "]),
    ("claude", ["claude"]),
    ("gemini", ["gemini", "palm"]),
    ("gemma", ["gemma"]),
    ("phi", ["phi-", "phi3", "phi4", "phi_"]),
    ("yi", ["yi-", "yi_", "yi1"]),
    ("deepseek", ["deepseek"]),
    ("falcon", ["falcon"]),
    ("command", ["command", "cohere", "c4ai", "aya"]),
    ("glm", ["glm", "chatglm"]),
    ("grok", ["grok"]),
    ("dbrx", ["dbrx"]),
    ("jamba", ["jamba", "jurassic"]),
    ("internlm", ["internlm"]),
    ("baichuan", ["baichuan"]),
    ("olmo", ["olmo"]),
    ("stablelm", ["stablelm", "stable-code"]),
    ("pythia", ["pythia", "dolly"]),
    ("mpt", ["mpt-"]),
    ("solar", ["solar"]),
    ("exaone", ["exaone"]),
    ("granite", ["granite"]),
    ("reka", ["reka"]),
    ("arctic", ["arctic"]),
    ("kimi", ["kimi", "moonshot"]),
    ("minimax", ["minimax"]),
    ("ernie", ["ernie"]),
    ("hunyuan", ["hunyuan"]),
    ("smollm", ["smollm"]),
]

ORG_DATASETS = {"mteb"}

# Dataset-name prefix -> per-model organization field of its meta.json
# (meta_org scheme).  "models" there is a list of {"name": ..., field: ...}
# (Epoch) or a dict {name: {field: ...}} (HELM).
META_ORG_FIELDS = {
    "epoch": "organization",
    "helm_": "creator_organization",
}

# meta_org scheme, rule 2: organization slugs (org_family) of one
# developer -> its label.  Epoch records Google's models as "Google
# DeepMind" (35, incl. "Google DeepMind,Google") and "Google" (Gemini 3.1
# Flash-Lite), Microsoft's as "Microsoft" (Phi-1.5 .. phi-3) and
# "Microsoft Research" (Phi-4).  HELM's labels are not affected.
ORG_ALIASES = {
    "google-deepmind": "google",
    "microsoft-research": "microsoft",
}

# meta_org scheme, rule 1: lowercased name substrings of fine-tunes,
# distillations and reproductions of another developer's base model ->
# the keyword family (KEYWORD_FAMILIES) of that base model.  Epoch:
# vicuna-13b-v1.1 (LMSYS), Stable Beluga 2 (Stability AI, a Llama 2 70B
# fine-tune), open_llama_7b (OpenLM Research's LLaMA reproduction),
# DeepSeek-R1-Distill-Qwen-14B/32B (DeepSeek, Qwen2.5 base); HELM
# Classic: Vicuna v1.3 7B/13B (LMSYS), Alpaca 7B (Stanford).
LINEAGE_KEYWORDS = [
    ("llama", ["vicuna", "alpaca", "stable beluga", "open_llama",
               "openllama", "distill-llama"]),
    ("qwen", ["distill-qwen"]),
]

# meta_org scheme, rules 1 and 3: keyword family -> candidate developer
# labels (rule-2 labels on the datasets that use them; the first one a
# model of the dataset carries wins, else the first).  Keyword families
# whose names also match other developers' models (gpt: GPT-J, Cerebras-
# GPT; pythia: Dolly) are left out: a model without an organization in
# such a family is "other".
BASE_DEVELOPERS = {
    "llama": ["meta-ai", "meta"],
    "qwen": ["alibaba", "qwen"],
    "gemini": ["google"],
    "gemma": ["google"],
    "deepseek": ["deepseek"],
    "glm": ["z-ai-zhipu-ai", "tsinghua"],
    "yi": ["01-ai"],
    "internlm": ["shanghai-ai-laboratory"],
    "stablelm": ["stability-ai"],
    "mistral": ["mistral-ai"],
    "phi": ["microsoft"],
    "falcon": ["technology-innovation-institute", "tii-uae"],
    "claude": ["anthropic"],
}


def meta_org_field(dataset: str):
    """Organization field of the meta_org scheme for `dataset`, or None."""
    for prefix, field in META_ORG_FIELDS.items():
        if dataset.startswith(prefix):
            return field
    return None


def family_scheme(dataset: str) -> str:
    """'org', 'meta_org' or 'keyword': the strategy for `dataset`."""
    if dataset in ORG_DATASETS or any(dataset.startswith(base)
                                      for base in ORG_DATASETS):
        return "org"
    if meta_org_field(dataset) is not None:
        return "meta_org"
    return "keyword"


def family_of(name: str, dataset: str) -> str:
    """Family of one model under the org or keyword scheme of `dataset`
    (the meta_org scheme needs the metadata: use assign_families)."""
    scheme = family_scheme(dataset)
    if scheme == "org":
        return name.split("/")[0].lower() if "/" in name else "other"
    if scheme == "meta_org":
        raise ValueError(f"{dataset}: families come from its meta.json; "
                         "use assign_families(..., data_dir=...)")
    return keyword_family(name)


def keyword_family(name: str) -> str:
    """Family of one model name under the keyword scheme."""
    lower = str(name).lower()
    # Special case: bare "yi" names like "Yi-34B" / "yi-6b" / "yi-large"
    if re.match(r"^yi[-_0-9]", lower):
        return "yi"
    for fam, keywords in KEYWORD_FAMILIES:
        if any(kw in lower for kw in keywords):
            return fam
    return "other"


def org_family(org) -> str:
    """Family label of an organization string (meta_org scheme): the slug
    of its first comma-separated entry, "other" if there is none."""
    if not org:
        return "other"
    first = str(org).split(",")[0]
    slug = re.sub(r"[^a-z0-9]+", "-", first.lower()).strip("-")
    return slug or "other"


def meta_organizations(dataset: str, model_names, data_dir) -> list:
    """Organization (str or None) of each model of `dataset`, read from
    data_dir/<dataset>.meta.json (meta_org scheme)."""
    import pathlib
    field = meta_org_field(dataset)
    if field is None:
        raise ValueError(f"{dataset}: no meta_org rule (META_ORG_FIELDS)")
    path = pathlib.Path(data_dir) / f"{dataset}.meta.json"
    with open(path) as f:
        models = json.load(f).get("models")
    if isinstance(models, list):
        by_name = {str(m.get("name")): m for m in models
                   if isinstance(m, dict)}
    elif isinstance(models, dict):
        by_name = {str(k): v for k, v in models.items()
                   if isinstance(v, dict)}
    else:
        raise ValueError(f"{path}: no per-model 'models' records")
    return [by_name.get(str(n), {}).get(field) for n in model_names]


def developer_label(org) -> str:
    """meta_org rule 2: the family label of an organization string,
    org_family merged by ORG_ALIASES ("other" if there is none)."""
    slug = org_family(org)
    return ORG_ALIASES.get(slug, slug)


def lineage_family(name: str):
    """meta_org rule 1: keyword family of the base model if `name` is a
    fine-tune / distillation / reproduction listed in LINEAGE_KEYWORDS,
    else None."""
    lower = str(name).lower()
    for fam, keywords in LINEAGE_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return fam
    return None


def meta_org_assignments(model_names, organizations):
    """[(label, rule)] per model under the meta_org scheme (module
    docstring): rule is "lineage", "organization", "keyword" (no
    organization: the keyword family's developer) or "other" (no
    organization and no developer for its keyword family).
    organizations: one str or None per model (meta_organizations)."""
    names = [str(n) for n in model_names]
    orgs = list(organizations)
    if len(orgs) != len(names):
        raise ValueError("one organization per model name expected")
    carried = {developer_label(o) for o in orgs if o} - {"other"}

    def developer(fam):
        cands = BASE_DEVELOPERS.get(fam)
        if not cands:
            return None
        return next((c for c in cands if c in carried), cands[0])

    out = []
    for name, org in zip(names, orgs):
        base = lineage_family(name)
        if base is not None and developer(base) is not None:
            out.append((developer(base), "lineage"))
        elif org and developer_label(org) != "other":
            out.append((developer_label(org), "organization"))
        else:
            dev = developer(keyword_family(name))
            out.append((dev, "keyword") if dev else ("other", "other"))
    return out


def assign_families(model_names, dataset: str, data_dir=None):
    """Return list of family labels, one per model name.

    The meta_org datasets (epoch, helm_*) read their organizations from
    data_dir/<dataset>.meta.json (default: path_config.data_dir()) and
    apply meta_org_assignments."""
    if family_scheme(dataset) == "meta_org":
        if data_dir is None:
            from path_config import data_dir as default_data_dir
            data_dir = default_data_dir()
        orgs = meta_organizations(dataset, model_names, data_dir)
        return [lab for lab, _ in meta_org_assignments(model_names, orgs)]
    return [family_of(str(n), dataset) for n in model_names]


if __name__ == "__main__":
    import pathlib
    import sys
    from collections import Counter

    import numpy as np

    CODE = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(CODE))
    from path_config import data_dir

    for ds in sys.argv[1:] or ["mmlu", "mteb", "merged", "benchpress"]:
        d = np.load(data_dir() / f"{ds}.matrix.npz", allow_pickle=True)
        fams = assign_families(d["model_names"], ds)
        counts = Counter(fams)
        M = len(fams)
        other = counts.get("other", 0)
        print(f"\n== {ds} ({M} models, {family_scheme(ds)} scheme, "
              f"{other} unmatched → 'other') ==")
        for fam, c in counts.most_common(25):
            print(f"  {fam:<22}{c}")
        if ds != "mmlu":
            unmatched = [str(n) for n, f in zip(d["model_names"], fams)
                         if f == "other"]
            if unmatched:
                print(f"  other: {', '.join(unmatched[:15])}"
                      + (" ..." if len(unmatched) > 15 else ""))
        if family_scheme(ds) == "meta_org":
            # Models not labelled by their own organization (rules 1, 3)
            names = [str(n) for n in d["model_names"]]
            orgs = meta_organizations(ds, names, data_dir())
            for (lab, rule), n, o in zip(meta_org_assignments(names, orgs),
                                         names, orgs):
                if rule in ("lineage", "keyword"):
                    print(f"  {rule:<9}{n} ({o or 'no organization'}) "
                          f"-> {lab}")
