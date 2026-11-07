"""Tests for model_families.py (synthetic metadata, < 5 s).

Covers the scheme per dataset name, the keyword and org rules (labels of
the paper's datasets and of olv1items / olv2sub / mteb2026 frozen by hash
where data/ is present), the organization slug, both meta.json layouts
(Epoch's list, HELM's dict), and the meta_org rules: one label per
developer (ORG_ALIASES), fine-tunes with their base model's developer
(LINEAGE_KEYWORDS), models without an organization with their keyword
family's developer (BASE_DEVELOPERS, resolved to the label the dataset
carries), and on the real Epoch / HELM Classic metadata the reviewed
assignments.  Also checks that a held-out family split
(eval_family_holdout.family_splits) never trains on a member of the
held-out family, lineage members included.
"""

import hashlib
import json
import pathlib

import numpy as np
import pytest

import model_families as mf
from path_config import data_dir

DATA = data_dir()


def _write_meta(path, models):
    path.write_text(json.dumps({"models": models}))


# ── Schemes and the unchanged rules ──────────────────────────────

def test_schemes():
    for ds in ["mmlu", "merged", "benchpress", "olv2sub", "olv1items"]:
        assert mf.family_scheme(ds) == "keyword", ds
    for ds in ["mteb", "mteb2026"]:
        assert mf.family_scheme(ds) == "org", ds
    for ds in ["epoch", "helm_lite", "helm_mmlu", "helm_classic"]:
        assert mf.family_scheme(ds) == "meta_org", ds
    with pytest.raises(ValueError):
        mf.family_of("Claude 2.1", "epoch")


def test_keyword_and_org_rules():
    names = ["meta-llama/Llama-2-7b", "Qwen/Qwen2-7B", "foo/bar", "Yi-34B",
             "claude-3-opus", "gpt-4o", "lmsys/vicuna-13b"]
    exp_kw = ["llama", "qwen", "other", "yi", "claude", "gpt", "llama"]
    for ds in ["mmlu", "merged", "benchpress", "olv2sub"]:
        assert mf.assign_families(names, ds) == exp_kw
    assert [mf.keyword_family(n) for n in names] == exp_kw
    assert mf.assign_families(names, "mteb") == [
        "meta-llama", "qwen", "foo", "other", "other", "other", "lmsys"]
    assert mf.assign_families(names, "mteb2026") == \
        mf.assign_families(names, "mteb")


# sha256 (first 16 hex) of "\n".join(labels) before the meta_org changes
FROZEN = {"mmlu": "71ec470d152e7108", "mteb": "879ca9c46fc65059",
          "merged": "20b2b6a81d9c5bf1", "benchpress": "c4be583de3147e0f",
          "olv2sub": "fd8ce00d13316260", "olv1items": "c59540cab190e5c0",
          "mteb2026": "985f2e0b8f78c36d"}


@pytest.mark.parametrize("ds", sorted(FROZEN))
def test_frozen_labels(ds):
    path = DATA / f"{ds}.matrix.npz"
    if not path.is_file():
        pytest.skip(f"no {path}")
    names = np.load(path, allow_pickle=True)["model_names"]
    lab = mf.assign_families(names, ds, data_dir=DATA)
    digest = hashlib.sha256("\n".join(lab).encode()).hexdigest()[:16]
    assert digest == FROZEN[ds]


def test_org_family_slug():
    assert mf.org_family("Mistral AI") == "mistral-ai"
    assert mf.org_family("DeepSeek,Peking University") == "deepseek"
    assert mf.org_family("Z.ai (Zhipu AI)") == "z-ai-zhipu-ai"
    assert mf.org_family(None) == "other"
    assert mf.org_family("") == "other"
    assert mf.org_family("!!!") == "other"
    assert mf.developer_label("Google DeepMind,Google") == "google"
    assert mf.developer_label("Google") == "google"
    assert mf.developer_label("Microsoft Research") == "microsoft"
    assert mf.developer_label("Meta") == "meta"


# ── meta_org rules ───────────────────────────────────────────────

def test_meta_org_layouts_and_rules(tmp_path):
    names = ["GPT-4", "Gemini 2.0 Pro", "Gemini 3.1 Flash-Lite", "PaLM 2-L",
             "Phi-2", "Phi-4", "Qwen2.5-72B", "Qwen2.5-Coder-3B",
             "DeepSeek-R1-Distill-Qwen-14B", "DeepSeek-V3", "Llama 3-8B",
             "vicuna-13b-v1.1", "Stable Beluga 2", "open_llama_7b",
             "Nemotron-4 15B", "internlm-7b", "RedPajama-INCITE-7B-Base",
             "missing-from-meta"]
    orgs = ["OpenAI", "Google DeepMind,Google", "Google", None, "Microsoft",
            "Microsoft Research", "Alibaba", None, "DeepSeek", "DeepSeek",
            "Meta AI", None, "Stability AI", None, "Nvidia", None, None]
    _write_meta(tmp_path / "epoch.meta.json",
                [{"name": n, "organization": o}
                 for n, o in zip(names, orgs)])
    got = mf.assign_families(names, "epoch", data_dir=tmp_path)
    assert got == ["openai", "google", "google", "google", "microsoft",
                   "microsoft", "alibaba", "alibaba", "alibaba", "deepseek",
                   "meta-ai", "meta-ai", "meta-ai", "meta-ai", "nvidia",
                   "shanghai-ai-laboratory", "other", "other"]
    rules = [r for _, r in mf.meta_org_assignments(
        names, mf.meta_organizations("epoch", names, tmp_path))]
    assert rules[3] == "keyword" and rules[8] == "lineage"
    assert rules[11] == "lineage" and rules[12] == "lineage"
    assert rules[16] == "other" and rules[0] == "organization"

    # HELM's dict layout; the lineage target is the label HELM carries
    helm = {"Vicuna v1.3 (7B)": "LMSYS", "Alpaca (7B)": "Stanford",
            "LLaMA (7B)": "Meta", "OPT (66B)": "Meta", "Qwen2 (72B)": "Qwen",
            "Palmyra X V3 (72B)": "Writer"}
    (tmp_path / "helm_x.meta.json").write_text(json.dumps({"models": {
        n: {"creator_organization": o} for n, o in helm.items()}}))
    assert mf.assign_families(list(helm), "helm_x", data_dir=tmp_path) == [
        "meta", "meta", "meta", "meta", "qwen", "writer"]


def test_candidate_resolution():
    # without a carried candidate the first BASE_DEVELOPERS entry is used
    out = mf.meta_org_assignments(["Qwen-1_8B", "Llama-7B-chat"],
                                  [None, None])
    assert out == [("alibaba", "keyword"), ("meta-ai", "keyword")]
    out = mf.meta_org_assignments(["Qwen-1_8B", "Qwen2 (7B)", "Alpaca (7B)",
                                   "LLaMA (7B)"],
                                  [None, "Qwen", "Stanford", "Meta"])
    assert [lab for lab, _ in out] == ["qwen", "qwen", "meta", "meta"]
    with pytest.raises(ValueError):
        mf.meta_org_assignments(["a", "b"], [None])
    # every BASE_DEVELOPERS / LINEAGE target is a keyword family
    kw = {fam for fam, _ in mf.KEYWORD_FAMILIES}
    assert set(mf.BASE_DEVELOPERS) <= kw
    assert {fam for fam, _ in mf.LINEAGE_KEYWORDS} <= set(mf.BASE_DEVELOPERS)


def test_real_epoch_and_helm_classic():
    if not (DATA / "epoch.matrix.npz").is_file():
        pytest.skip("no epoch matrix")
    names = [str(n) for n in np.load(DATA / "epoch.matrix.npz",
                                     allow_pickle=True)["model_names"]]
    lab = dict(zip(names, mf.assign_families(names, "epoch", DATA)))
    labels = set(lab.values())
    assert not labels & {"google-deepmind", "microsoft-research"}
    assert [n for n, f in lab.items() if f == "other"] == [
        "RedPajama-INCITE-7B-Base"]
    for n in ["PaLM 2-L", "PaLM 2-M", "PaLM 2-S", "Gemini 3.1 Flash-Lite"]:
        assert lab[n] == "google", n
    for n in ["CodeQwen1.5-7B", "Qwen-1_8B", "Qwen2.5-Coder-0.5B",
              "Qwen2.5-Coder-3B", "Qwen2.5-Coder-14B",
              "DeepSeek-R1-Distill-Qwen-14B", "DeepSeek-R1-Distill-Qwen-32B"]:
        assert lab[n] == "alibaba", n
    for n in ["vicuna-13b-v1.1", "open_llama_7b", "Stable Beluga 2"]:
        assert lab[n] == "meta-ai", n
    assert lab["chatglm2-6b"] == "z-ai-zhipu-ai"
    assert lab["DeepSeek-Coder-V2-Lite-Base"] == "deepseek"
    assert lab["Yi-9B"] == "01-ai"
    assert sum(f == "microsoft" for f in lab.values()) == 6
    assert sum(f == "google" for f in lab.values()) == 39

    if not (DATA / "helm_classic.matrix.npz").is_file():
        return
    names = [str(n) for n in np.load(DATA / "helm_classic.matrix.npz",
                                     allow_pickle=True)["model_names"]]
    lab = dict(zip(names, mf.assign_families(names, "helm_classic", DATA)))
    assert not set(lab.values()) & {"lmsys", "stanford"}
    for n in ["Vicuna v1.3 (13B)", "Vicuna v1.3 (7B)", "Alpaca (7B)"]:
        assert lab[n] == "meta", n
    assert "other" not in lab.values()


def test_family_split_never_trains_on_held_out_members(tmp_path):
    from eval_family_holdout import family_splits
    names = ["LLaMA (7B)", "Llama 2 (13B)", "Vicuna v1.3 (7B)", "Alpaca (7B)",
             "GPT-4", "Claude 2", "Qwen2 (7B)", "Mistral 7B"]
    orgs = ["Meta", "Meta", "LMSYS", "Stanford", "OpenAI", "Anthropic",
            "Qwen", "Mistral AI"]
    (tmp_path / "helm_y.meta.json").write_text(json.dumps({"models": {
        n: {"creator_organization": o} for n, o in zip(names, orgs)}}))
    fams = np.array(mf.assign_families(names, "helm_y", data_dir=tmp_path))
    held = ["meta", "openai"]
    for s in family_splits(fams, held, n_repeats=2):
        assert set(fams[s.val_idx]) == {s.family}
        assert s.family not in set(fams[s.train_idx])
        if s.family == "meta":
            assert sorted(s.val_idx.tolist()) == [0, 1, 2, 3]
