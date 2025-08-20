# src/keyword_extractor.py
from __future__ import annotations
from typing import Dict, List, Set, Optional
import os
import re
import math
import numpy as np

# --- YAKE ---
import yake

# --- Optional semantic pieces ---
try:
    from keybert import KeyBERT
except Exception:
    KeyBERT = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# --- spaCy ---
try:
    import spacy
    from spacy.tokens import Doc
except Exception:
    spacy = None
    Doc = None

# -------------------------
# Constants / Helpers
# -------------------------
_STOPLIKE: Set[str] = {
    "the","a","an","and","or","of","for","to","in","on","by","with","from",
    "at","as","is","are","be","being","been","that","this","these","those"
}

_SINGLETON_BLACKLIST: Set[str] = {
    "page","pages","figure","fig","table","chapter","section","slide","edition",
    "overview","general","objectives","history","introduction","summary","copy",
    "copyright","license","licensing","contents"
}

_PHRASE_DOWNRANK: Set[str] = {
    "page","pages","figure","table","edition","general","chapter","section","slide",
    "copy","copyright","license","licensing","contents","tm","©","®"
}

_SEMANTIC_BLACKLIST = [
    "introduction", "summary", "objectives", "conclusion", "overview",
    "table of contents", "chapter", "figure", "slide", "appendix",
    "acknowledgements", "references", "bibliography", "index"
]

_PUNCT_RE = re.compile(r"[^\w\s\-\/&]+")

def _clean_phrase(p: str) -> str:
    p = _PUNCT_RE.sub(" ", p)
    p = re.sub(r"\s+", " ", p).strip()
    return p

def _tokenize(p: str) -> List[str]:
    return [t for t in _clean_phrase(p).split() if t]

def _titleish(s: str) -> str:
    parts = s.split()
    out = []
    for w in parts:
        if w.isupper() and len(w) <= 6:
            out.append(w)
        else:
            out.append(w.capitalize())
    return " ".join(out)

def _normalize_scores(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vals = np.array(list(d.values()), dtype=float)
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if vmax - vmin < 1e-12:
        return {k: 0.5 for k in d}
    return {k: (v - vmin) / (vmax - vmin) for k, v in d.items()}

# ------------------------
# Safe model loads
# ------------------------
def _safe_load_spacy():
    if spacy is None:
        return None
    for name in ("en_core_web_sm", "en_core_web_md"):
        try:
            return spacy.load(name)
        except Exception:
            continue
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp

def _safe_load_embedder() -> Optional[SentenceTransformer]:
    if SentenceTransformer is None:
        return None
    model_name = os.getenv("ATLEARN_EMBED_MODEL", "all-mpnet-base-v2")
    try:
        return SentenceTransformer(model_name)
    except Exception:
        return None

_NLP = _safe_load_spacy()
_EMBEDDER = _safe_load_embedder()
_KW_MODEL = None
if KeyBERT is not None and _EMBEDDER is not None:
    try:
        _KW_MODEL = KeyBERT(model=_EMBEDDER)
    except Exception:
        _KW_MODEL = None

# ------------------------
# Candidate generation
# ------------------------
def _extract_yake(text: str, k: int) -> Dict[str, float]:
    extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=k, features=None)
    pairs = extractor.extract_keywords(text)
    out: Dict[str, float] = {}
    for kw, score in pairs:
        kw = _clean_phrase(kw).lower()
        if kw and len(kw.split()) <= 5:
            out[kw] = float(score)
    return out

def _extract_keybert(text: str, k: int) -> Dict[str, float]:
    if _KW_MODEL is None:
        return {}
    try:
        pairs = _KW_MODEL.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            use_mmr=True,
            diversity=0.7,
            top_n=k,
        )
    except Exception:
        return {}
    out: Dict[str, float] = {}
    for kw, score in pairs:
        kw = _clean_phrase(kw).lower()
        if kw and len(kw.split()) <= 5:
            out[kw] = float(score)
    return out

def _extract_noun_chunks(text: str, k: int) -> Set[str]:
    out: Set[str] = set()
    if _NLP is None:
        return out
    try:
        doc = _NLP(text)
        for chunk in getattr(doc, "noun_chunks", []):
            kw = _clean_phrase(chunk.text).lower()
            toks = _tokenize(kw)
            if not toks:
                continue
            if 1 <= len(toks) <= 5 and not (len(toks)==1 and toks[0] in _SINGLETON_BLACKLIST):
                out.add(kw)
            if len(out) >= k:
                break
    except Exception:
        pass
    return out

def _extract_ner(text: str) -> Set[str]:
    keep: Set[str] = set()
    if _NLP is None or not hasattr(_NLP, "pipe_names"):
        return keep
    try:
        doc = _NLP(text)
        for ent in doc.ents:
            if ent.label_ in {"ORG","PRODUCT","GPE","PERSON","NORP","EVENT","WORK_OF_ART","LAW"}:
                kw = _clean_phrase(ent.text).lower()
                if kw and len(kw.split()) <= 5:
                    keep.add(kw)
    except Exception:
        pass
    return keep

# ------------------------
# Scoring & filtering
# ------------------------
def _downrank_score(phrase: str) -> float:
    toks = _tokenize(phrase)
    if not toks:
        return 0.0
    penalty = 0.0
    for t in toks:
        if t in _PHRASE_DOWNRANK:
            penalty += 0.15
    if len(toks) == 1 and toks[0] in _SINGLETON_BLACKLIST:
        penalty += 0.4
    if len(toks) in (2, 3):
        penalty -= 0.08
    return max(0.0, min(0.6, penalty))

def _fuse_scores(
    text: str,
    yake_raw: Dict[str, float],
    kb_raw: Dict[str, float],
    noun_chunks: Set[str],
    ner_keep: Set[str]
) -> Dict[str, float]:
    yake_inv = {p: 1.0 / (s + 1e-6) for p, s in yake_raw.items()}
    yake_n = _normalize_scores(yake_inv)
    kb_n = _normalize_scores(kb_raw)
    cand: Set[str] = set(yake_raw) | set(kb_raw) | set(noun_chunks) | set(ner_keep)
    fused: Dict[str, float] = {}
    for p in cand:
        base = 0.0
        base += 0.45 * yake_n.get(p, 0.0)
        base += 0.55 * kb_n.get(p, 0.0)
        if p in ner_keep:
            base += 0.15
        base -= _downrank_score(p)
        occ = text.lower().count(p)
        base += min(0.08, math.log1p(occ) * 0.03)
        fused[p] = base
    return {p: s for p, s in fused.items() if s > 0.05}

# ------------------------
# Deduplication / semantic cleaning
# ------------------------
def _lemma_canon(phrase: str) -> str:
    if _NLP is None:
        return phrase.lower()
    try:
        doc = _NLP(phrase)
        parts = [t.lemma_.lower() for t in doc if t.text.strip() and t.text.lower() not in _STOPLIKE]
        return " ".join(parts).strip() or phrase.lower()
    except Exception:
        return phrase.lower()

def _embed(texts: List[str]) -> np.ndarray:
    if _EMBEDDER is None or not texts:
        return np.zeros((0, 8), dtype=np.float32)
    return _EMBEDDER.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def _deduplicate(phrases_ranked: List[str], sim_threshold: float = 0.84) -> List[str]:
    if not phrases_ranked:
        return []
    kept: List[str] = []
    seen: Set[str] = set()
    for p in phrases_ranked:
        c = _lemma_canon(p)
        if c in seen:
            continue
        if any(c in _lemma_canon(q) or _lemma_canon(q) in c for q in kept):
            continue
        kept.append(p)
        seen.add(c)
    if _EMBEDDER:
        limit = min(len(kept), 80)
        subset = kept[:limit]
        embs = _embed(subset)
        final: List[str] = []
        for i, p in enumerate(subset):
            if not final:
                final.append(p)
                continue
            idxs = [subset.index(x) for x in final]
            sims = np.dot(embs[i], embs[idxs].T)
            if np.max(sims) < sim_threshold:
                final.append(p)
        if limit < len(kept):
            for p in kept[limit:]:
                if p not in final:
                    final.append(p)
        return final
    return kept

def _semantic_clean(phrases: List[str], sim_thresh: float = 0.75) -> List[str]:
    if not _EMBEDDER or not phrases:
        return phrases
    cleaned = []
    phrase_embs = _EMBEDDER.encode(phrases, convert_to_numpy=True, normalize_embeddings=True)
    blacklist_embs = _EMBEDDER.encode(_SEMANTIC_BLACKLIST, convert_to_numpy=True, normalize_embeddings=True)
    for i, p in enumerate(phrases):
        sims = np.dot(phrase_embs[i], blacklist_embs.T)
        if np.max(sims) < sim_thresh:
            cleaned.append(p)
    return cleaned

# ------------------------
# Hierarchy building
# ------------------------
def _head_noun(phrase: str) -> str:
    toks = _tokenize(phrase)
    if not toks:
        return ""
    if _NLP is not None:
        try:
            doc = _NLP(phrase)
            nouns = [t.lemma_.lower() for t in doc if t.pos_ in ("NOUN","PROPN")]
            if nouns:
                return nouns[-1]
        except Exception:
            pass
    for t in reversed(toks):
        if t.lower() not in _STOPLIKE:
            return t.lower()
    return toks[-1].lower()

def _build_hierarchy(phrases: List[str], top_n: int) -> Dict[str, List[str]]:
    heads_scores: Dict[str, float] = {}
    for p in phrases:
        h = _head_noun(p)
        if not h:
            continue
        heads_scores[h] = max(heads_scores.get(h, 0.0), 1.0)
    head_to_phrases: Dict[str, List[str]] = {}
    for p in phrases:
        h = _head_noun(p)
        if not h:
            continue
        head_to_phrases.setdefault(h, []).append(p)
    ordered_heads = sorted(head_to_phrases.keys(), key=lambda h: (-len(head_to_phrases[h]), h))
    result: Dict[str, List[str]] = {}
    for h in ordered_heads[:top_n]:
        subs = head_to_phrases[h]
        subs_sorted = sorted(
            subs,
            key=lambda s: (h not in s.lower(), abs(len(_tokenize(s)) - 3), len(s))
        )
        result[_titleish(h)] = [_titleish(s) for s in subs_sorted[:max(5, min(10, len(subs_sorted)))]]
    return result

# ------------------------
# Public API
# ------------------------
def extract_keywords_hierarchical(text: str, top_n: int = 10) -> Dict[str, List[str]]:
    if not isinstance(text, str) or not text.strip():
        return {}
    text = text.strip()
    if len(text) > 120_000:
        text = text[:120_000]

    # Candidate generation
    k_pool = max(40, top_n * 6)
    yake_raw = _extract_yake(text, k_pool)
    kb_raw = _extract_keybert(text, k_pool // 2)
    noun_chunks = _extract_noun_chunks(text, k_pool)
    ner_keep = _extract_ner(text)

    # Scoring + filtering
    fused = _fuse_scores(text, yake_raw, kb_raw, noun_chunks, ner_keep)
    if not fused:
        return {}

    ranked = [p for p, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]
    filtered = [p for p in ranked if 1 <= len(_tokenize(p)) <= 5 and not (len(_tokenize(p))==1 and _tokenize(p)[0] in _SINGLETON_BLACKLIST)]

    # Deduplicate + semantic cleaning
    deduped = _deduplicate(filtered, sim_threshold=0.84)
    deduped = _semantic_clean(deduped, sim_thresh=0.75)

    top_phrases = deduped[:max(60, top_n * 8)]
    return _build_hierarchy(top_phrases, top_n)
