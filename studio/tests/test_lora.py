"""LoRA instrumentale : fusion des deltas, résolution du fichier, réécriture des paroles, bascule dans le moteur."""
from contextlib import contextmanager

import pytest
import torch
from torch import nn

from studio import lora


# ---------------------------------------------------------------- modèle jouet
class _Attn(nn.Module):
    def __init__(self, h, kv):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = nn.Linear(h, h, bias=False), nn.Linear(h, kv, bias=False), nn.Linear(h, kv, bias=False), nn.Linear(h, h, bias=False)


class _Mlp(nn.Module):
    def __init__(self, h, inter):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = nn.Linear(h, inter, bias=False), nn.Linear(h, inter, bias=False), nn.Linear(inter, h, bias=False)


class _Layer(nn.Module):
    def __init__(self, h, kv, inter):
        super().__init__()
        self.self_attn, self.mlp = _Attn(h, kv), _Mlp(h, inter)


class _Backbone(nn.Module):
    def __init__(self, n, h=8, kv=4, inter=16):
        super().__init__()
        self.layers = nn.ModuleList(_Layer(h, kv, inter) for _ in range(n))


class TinyModel(nn.Module):
    def __init__(self, n=2, dtype=torch.bfloat16):
        super().__init__()
        torch.manual_seed(0)
        self.model = _Backbone(n)
        self.to(dtype)


def make_tensors(model, rank=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for i, layer in enumerate(model.model.layers):
        for block, names in lora.PROJECTIONS:
            for n in names:
                w = getattr(getattr(layer, block), n).weight
                out[f"layers.{i}.{block}.{n}.lora_A"] = torch.randn(rank, w.shape[1], generator=g).to(torch.bfloat16)
                out[f"layers.{i}.{block}.{n}.lora_B"] = torch.randn(w.shape[0], rank, generator=g).to(torch.bfloat16)
    return out


def snapshot(model):
    return {k: v.detach().clone() for k, v in model.named_parameters()}


# ---------------------------------------------------------------- fusion
def test_merge_adds_scaled_delta_to_every_projection():
    model = TinyModel(n=2)
    before = snapshot(model)
    tensors = make_tensors(model)
    assert lora.merge(model, tensors, scale=0.5) == 2 * 7
    for i, layer in enumerate(model.model.layers):
        for block, names in lora.PROJECTIONS:
            for n in names:
                key = f"layers.{i}.{block}.{n}"
                a, b = tensors[f"{key}.lora_A"].float(), tensors[f"{key}.lora_B"].float()
                expected = (before[f"model.layers.{i}.{block}.{n}.weight"].float() + 0.5 * (b @ a)).to(torch.bfloat16)
                got = getattr(getattr(layer, block), n).weight
                assert torch.equal(got, expected), key
    assert lora.describe(tensors) == {"layers": 2, "rank": 3, "tensors": 28}


def test_merge_refuses_missing_tensor_shape_mismatch_and_fp8():
    model = TinyModel(n=1)
    tensors = make_tensors(model)
    incomplete = {k: v for k, v in tensors.items() if not k.endswith("mlp.down_proj.lora_B")}
    with pytest.raises(lora.LoRAError, match="manquant"):
        lora.merge(model, incomplete)
    wrong = dict(tensors)
    wrong["layers.0.self_attn.q_proj.lora_A"] = torch.zeros(3, 5, dtype=torch.bfloat16)
    with pytest.raises(lora.LoRAError, match="Formes incompatibles"):
        lora.merge(model, wrong)
    object.__setattr__(model, "_yue2_fp8_originals", {"x": 1})
    with pytest.raises(lora.LoRAError, match="FP8"):
        lora.merge(model, tensors)


def test_load_tensors_and_resolve_file(tmp_path):
    from safetensors.torch import save_file
    model = TinyModel(n=1)
    tensors = make_tensors(model)
    path = tmp_path / "lora.safetensors"
    save_file(tensors, str(path), metadata={"rank": "3"})
    loaded = lora.load_tensors(path)
    assert set(loaded) == set(tensors) and torch.equal(loaded["layers.0.mlp.up_proj.lora_B"], tensors["layers.0.mlp.up_proj.lora_B"])
    # fichier valide mais sans lora_A/lora_B (ex. variante ComfyUI fusionnée)
    save_file({"text_encoders.model.layers.0.qkv_proj.lora_up": torch.zeros(2, 2)}, str(tmp_path / "comfy.safetensors"))
    with pytest.raises(lora.LoRAError, match="lora_A"):
        lora.load_tensors(tmp_path / "comfy.safetensors")
    (tmp_path / "broken.safetensors").write_bytes(b"nope")
    with pytest.raises(lora.LoRAError, match="illisible"):
        lora.load_tensors(tmp_path / "broken.safetensors")
    # résolution : fichier local, dossier + nom, dossier sans le fichier, dépôt vide
    assert lora.resolve_file(str(path)) == path
    assert lora.resolve_file(str(tmp_path), "lora.safetensors") == path
    with pytest.raises(lora.LoRAError, match="introuvable"):
        lora.resolve_file(str(tmp_path), "absent.safetensors")
    with pytest.raises(lora.LoRAError, match="Aucun dépôt"):
        lora.resolve_file("")


# ---------------------------------------------------------------- paroles → structure
@pytest.mark.parametrize("lyrics, expected", [
    ("[Verse]\nla la la\nli li\n\n[Chorus]\nsing it\n", "[verse]\n[chorus]"),
    ("[Intro]\n\n[Interlude]\n\n[Outro]", "[intro]\n[bridge]\n[outro]"),
    ("", "[instrumental]"),
    ("[Instrumental]", "[instrumental]"),
    ("[instrumental]\n[instrumental]", "[instrumental]"),
    ("[Verse 0:15-0:45]\n[Chorus 0:45–1:10]", "[verse 0:15-0:45]\n[chorus 0:45-1:10]"),
    ("[Verse 2]\n[Pre-Chorus]\n[Refrain]\n[Hook]", "[verse]\n[pre-chorus]\n[chorus]\n[chorus]"),
    ("[Guitar solo, distorted]\n[Break]", "[bridge]"),
    ("[Couplet]\n[Pont]\n[Final]", "[verse]\n[bridge]\n[outro]"),
    ("just words\nno tags", "[instrumental]"),
])
def test_instrumental_lyrics(lyrics, expected):
    assert lora.instrumental_lyrics(lyrics) == expected


# ---------------------------------------------------------------- bascule dans le moteur
class LoRAPipe:
    """Pipeline factice : un modèle jouet rechargeable, un _status qui journalise les étapes."""

    def __init__(self, backend="torch"):
        self.backend, self.stages, self.loads = backend, [], 0
        self._model = None
        self._load_model()

    def _load_model(self):
        if self._model is None:
            self._model = TinyModel(n=2)     # même graine : recharger redonne exactement les poids d'origine
            self.loads += 1
        return self._model

    @contextmanager
    def _status(self, label, **kw):
        self.stages.append(label)
        yield None


@pytest.fixture
def lora_file(tmp_path):
    from safetensors.torch import save_file
    tensors = make_tensors(TinyModel(n=2))
    path = tmp_path / "inst.bf16.safetensors"
    save_file(tensors, str(path), metadata={"rank": "3"})
    return path, tensors


def test_engine_applies_and_removes_lora_between_jobs(engine, make_job, lora_file, monkeypatch):
    path, tensors = lora_file
    monkeypatch.setattr(engine.settings, "instrumental_lora", str(path))
    monkeypatch.setattr(engine.settings, "instrumental_lora_scale", 1.0)
    pipe = LoRAPipe()
    original = snapshot(pipe._model)
    song = make_job("chanson", status="queued", audio=False)
    inst = make_job("instru", status="queued", audio=False)
    inst.instrumental = True
    # chanson : rien à faire
    assert engine._ensure_lora(pipe, song) is None and pipe.stages == [] and pipe.loads == 1
    # instrumental : fusion, une seule fois
    info = engine._ensure_lora(pipe, inst)
    assert info["merged_linears"] == 14 and info["rank"] == 3 and info["scale"] == 1.0 and info["repo"] == str(path)
    assert pipe.stages == ["Applying instrumental LoRA"]
    w = pipe._model.model.layers[0].self_attn.q_proj.weight
    assert not torch.equal(w, original["model.layers.0.self_attn.q_proj.weight"])
    assert engine._ensure_lora(pipe, inst) == info and pipe.stages == ["Applying instrumental LoRA"]   # déjà en place
    # retour à une chanson : le modèle est rechargé (poids d'origine exacts), pas soustrait
    assert engine._ensure_lora(pipe, song) is None
    assert pipe.stages[-1] == "Reloading base model" and pipe.loads == 2
    assert torch.equal(pipe._model.model.layers[0].self_attn.q_proj.weight, original["model.layers.0.self_attn.q_proj.weight"])
    assert getattr(pipe, "_studio_lora", None) is None
    # nouvelle intensité : fusion sur le modèle de base déjà rechargé (pas de rechargement supplémentaire)
    monkeypatch.setattr(engine.settings, "instrumental_lora_scale", 0.5)
    info2 = engine._ensure_lora(pipe, inst)
    assert info2["scale"] == 0.5 and pipe.loads == 2 and pipe.stages[-1] == "Applying instrumental LoRA"
    # changer l'intensité alors que la LoRA est en place : rechargement puis nouvelle fusion
    monkeypatch.setattr(engine.settings, "instrumental_lora_scale", 0.7)
    info3 = engine._ensure_lora(pipe, inst)
    assert info3["scale"] == 0.7 and pipe.loads == 3 and pipe.stages[-2:] == ["Reloading base model", "Applying instrumental LoRA"]


def test_engine_lora_errors_are_explicit(engine, make_job, lora_file, monkeypatch):
    path, _ = lora_file
    inst = make_job("instru", status="queued", audio=False)
    inst.instrumental = True
    monkeypatch.setattr(engine.settings, "instrumental_lora", str(path))
    with pytest.raises(lora.LoRAError, match="vllm"):
        engine._ensure_lora(LoRAPipe(backend="vllm"), inst)
    monkeypatch.setattr(engine.settings, "instrumental_lora", str(path.parent / "absent.safetensors"))
    pipe = LoRAPipe()
    with pytest.raises(lora.LoRAError):
        engine._ensure_lora(pipe, inst)
    assert getattr(pipe, "_studio_lora", None) is None     # un échec ne laisse pas croire que la LoRA est en place
