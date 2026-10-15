"""Typed configuration for a full pipeline run.

The whole point: one ``RunConfig`` object is the source of truth for every
parameter the pipeline reads. Stages cannot disagree on shared fields
because they read the same struct (e.g. ``cfg.aut.prompts.*`` is consumed
by both ``steer_collect`` and ``aut_infer``).

Load from YAML with :func:`load_run_config`; unknown keys raise — that's how
the old ``run_aut_inference._format_user_prompt`` bug stayed hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Type, Union, get_args, get_origin


# Re-export at module level so callers can write `from bridge_steering.config import ConfigError`.
class ConfigError(ValueError):
    """Raised by :func:`load_run_config` for any validation failure."""


# Re-exported so YAMLs / users have one place to look up allowed values.
PromptSubject = Literal["object_only", "full_instruction"]
MacgyverPromptStyle = Literal["default", "novelty_instruct", "mechanism_style"]


# ---------------------------------------------------------------------------
# AUT prompt config — coupled by default, explicit decoupling only.
# ---------------------------------------------------------------------------


@dataclass
class AUTPromptOverride:
    """Either pin a prompt by id, or pick by subject — never both."""

    prompt_id: Optional[str] = None
    subject: Optional[PromptSubject] = None


@dataclass
class AUTPrompts:
    default_subject: PromptSubject = "object_only"
    allow_decoupling: bool = False
    decoupling_reason: Optional[str] = None
    steering: Optional[AUTPromptOverride] = None
    inference: Optional[AUTPromptOverride] = None


@dataclass
class AUTConfig:
    abcd_data: Path
    prompts: AUTPrompts = field(default_factory=AUTPrompts)
    eval_seed: int = 7
    eval_holdout: int = 40
    # Learning-curve knob: cap the steering-training set to the first K items
    # of split.train_items after the eval split has been carved out. None =
    # use all train items (legacy behaviour). Subsetting is deterministic and
    # monotonic: train_subset_size=10 is a prefix of train_subset_size=20 etc.
    train_subset_size: Optional[int] = None


# ---------------------------------------------------------------------------
# Steering / decoding / external services
# ---------------------------------------------------------------------------


@dataclass
class SteeringConfig:
    # Where the contrastive "B" (bridge) completions come from when
    # collecting activations. Names registered in
    # bridge_steering.steering.completion_sources.SOURCE_REGISTRY (e.g. "fixed",
    # "multi_b_concat", "cross_object_train_b").
    bridge_source: str = "fixed"
    # Which extractor turns B/D activations into a steering vector — see
    # bridge_steering.steering.extractors.EXTRACTOR_REGISTRY (e.g. "mean_diff",
    # "b_only", "b_perp").
    method: str = "mean_diff"
    # Which ABCD field is the "D" (default / banal) completion.
    default_source: str = "D_banal"
    probe_layers: list[int] = field(default_factory=lambda: list(range(10, 25)))
    window: int = 0
    collect_span: Literal["assistant_full", "assistant_body"] = "assistant_full"
    steer_target: Literal["residual", "attn_out"] = "residual"
    steer_scale: Literal["fixed", "relative_norm"] = "fixed"
    method_kwargs: dict = field(default_factory=dict)
    source_kwargs: dict = field(default_factory=dict)


@dataclass
class OlsonConfig:
    """Olson-style (prompt-induced) creativity steering.

    Reproduces the recipe of Olson et al. (2024) — arXiv:2412.06060 — adapted
    to AUT via instruction-prefix prompts. The model generates a response
    under a *creative-instructed* prompt and under a *boring-instructed*
    prompt; ``μ_creative − μ_boring`` over the first ``window`` generated
    tokens at each probe layer is the steering vector.

    Activation collection happens in :mod:`bridge_steering.stages.olson_collect`;
    its output schema is identical to :mod:`bridge_steering.stages.steer_collect`'s,
    so the downstream ``steer_extract`` stage consumes Olson activations
    unchanged. ``steer_extract`` selects the input path based on whether
    ``cfg.olson`` is set.

    Structural difference vs. ``cfg.steering``: bridge / CAA-style contrasts
    hold the prompt fixed and vary the teacher-forced completion. Olson holds
    the input item fixed and varies *the prompt*, then reads activations
    under the model's own generation. This is captured here by carrying two
    prompt ids instead of a single completion source.
    """

    creative_prompt_id: str = "aut_olson_creative_v1"
    boring_prompt_id: str = "aut_olson_boring_v1"

    probe_layers: list[int] = field(default_factory=lambda: list(range(10, 25)))
    # Number of new tokens the model generates under each prompt before
    # activations are read.
    max_new_tokens: int = 96
    # First N generated-token positions averaged per probe layer; 0 means
    # "average over all generated tokens" — matches the production bridge
    # setting (window=0, collect_span=assistant_full). Set > 0 only to cap
    # to a fixed window if you want a length-controlled ablation.
    window: int = 0
    # Decoding for the per-side generations.
    gen_temperature: float = 0.7
    gen_top_p: Optional[float] = None
    # Seed for the generation calls — kept separate from the run seed so the
    # steering vector is reproducible across pipeline reruns.
    gen_seed: Optional[int] = 42
    # Hook target / scale — must match what ``aut_steering`` will apply
    # at inference time. Default = same as ``SteeringConfig``.
    steer_target: Literal["residual", "attn_out"] = "residual"
    steer_scale: Literal["fixed", "relative_norm"] = "fixed"


@dataclass
class DecodingConfig:
    temperature: float = 0.0
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    max_new_tokens: int = 384
    gen_seed: Optional[int] = None
    num_inferences: int = 1
    batch_size: int = 8


@dataclass
class SteeringInferenceConfig:
    """Steered method args used at inference. ``vectors`` may be filled in
    from the steer stage's output if left ``None``."""

    vectors: Optional[Path] = None
    alphas: list[float] = field(default_factory=lambda: [1.0])
    layers: Optional[list[int]] = None
    steer_layers_group: Optional[list[int]] = None
    steer_mode: Literal[
        "all_new_tokens", "first_k_assistant_tokens", "last_prompt_only", "all"
    ] = "all_new_tokens"
    k_assist: int = 16
    steer_start_offset: int = 0
    include_baseline: bool = True
    # When False, skip the fixed-vector SteeredMethod variants (alphas still
    # supply the dose for tau methods). Lets a run evaluate ONLY tau sampling.
    include_fixed_steered: bool = True
    # Per-call direction sampling (TauSteeredMethod). One tau method is built
    # per (alpha, tau) combo when `taus` is non-empty; `distribution` must
    # point at fit_steering_distribution.py output.
    distribution: Optional[Path] = None
    taus: list[float] = field(default_factory=list)
    tau_rank: int = 16


# ---------------------------------------------------------------------------
# Judge backends — vLLM (self-hosted) OR API (cloud endpoint).
# Both speak the same OpenAI-compatible chat-completions protocol, so
# stage code is identical; only the client construction differs.
# ---------------------------------------------------------------------------


@dataclass
class VllmJudgeBackend:
    """Self-hosted vLLM-served judge model.

    The vLLM server is started by ``managed_vllm`` (or assumed already running
    at ``base_url``). API key is a dummy string — vLLM ignores it.
    """

    type: Literal["vllm"] = "vllm"
    base_url: str = "http://localhost:8000/v1"
    # Repo / quantized id of the model the vLLM server is hosting.
    model: str = "Qwen/Qwen2.5-32B-Instruct-AWQ"


@dataclass
class ApiJudgeBackend:
    """Cloud judge via an OpenAI-compatible chat-completions endpoint.

    Works with OpenAI, Anthropic-via-OpenAI-compat, DeepInfra, Together,
    Fireworks, Groq, etc. Set ``base_url`` to the provider's v1 endpoint.

    The API key is read from the env var named in ``api_key_env`` so it
    never lands in the YAML or the manifest.
    """

    type: Literal["api"] = "api"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    # Env var name (NOT the value!) — manifest records the var name only.
    api_key_env: str = "OPENAI_API_KEY"
    # Optional org / project headers (OpenAI; ignored elsewhere).
    organization: Optional[str] = None
    project: Optional[str] = None


@dataclass
class AzureApiJudgeBackend:
    """Azure OpenAI judge via API key auth.

    ``model`` is the Azure **deployment name** (e.g. ``judge-gpt-4o-mini``), not
    the underlying model id like ``gpt-4o-mini``. Endpoint, key, and api version
    are read from the env vars named below so no secrets land in the YAML.
    """

    type: Literal["azure"] = "azure"
    # Azure DEPLOYMENT NAME — passed as ``model=`` to chat.completions.create.
    model: str = "gpt-4o-mini"
    api_key_env: str = "AZURE_OPENAI_API_KEY"
    endpoint_env: str = "AZURE_OPENAI_ENDPOINT"
    api_version_env: str = "OPENAI_API_VERSION"


# Type alias for readability — Pyright wants the inline Union on the field.
JudgeBackend = Union[VllmJudgeBackend, ApiJudgeBackend, AzureApiJudgeBackend]


@dataclass
class JudgeConfig:
    """Backend-agnostic judge settings.

    ``backend`` selects vLLM vs. API. Everything else applies to both.
    """

    backend: Union[VllmJudgeBackend, ApiJudgeBackend, AzureApiJudgeBackend]
    prompt_id: str = ""
    temperature: float = 0.0
    max_concurrent: int = 256   # API providers may need a much smaller value
    max_retries: int = 3
    timeout_s: float = 60.0
    max_tokens: int = 400
    # OpenAI/Azure ``seed`` for best-effort determinism. Same prompt + same
    # seed usually returns the same output; not guaranteed bit-exact (GPU
    # batching can break ties differently across runs), but it removes the
    # bulk of variance when a model forces temperature > 0 (e.g. gpt-5).
    seed: int = 42


@dataclass
class OCSAIConfig:
    url: str = "https://openscoring.du.edu/llm"
    model: str = "ocsai-davinci3"
    batch_size: int = 32
    max_attempts: int = 5
    retry_backoff_s: float = 2.0
    # Per-batch HTTP read timeout. The newer ``ocsai2-paper`` (gpt-4.1-mini
    # based) is noticeably slower per item than the legacy ``davinci3``;
    # the default needs to cover ``batch_size`` requests on the slowest
    # supported model. 120s is enough for a 32-row batch on ocsai2-paper.
    timeout_s: float = 120.0
    # Number of batches scored in parallel via ThreadPoolExecutor. Set to 1
    # for deterministic sequential execution (tests, debugging).
    max_concurrency: int = 4


@dataclass
class MacGyverInferenceConfig:
    prompt_style: MacgyverPromptStyle = "default"
    subset: str = "solvable_unconventional"
    max_items: Optional[int] = None


@dataclass
class DenialConfig:
    """Denial-prompting (Lu et al. 2024) — iterative constraint-stacking.

    Round 0 uses the standard task prompt; rounds 1..rounds inject an
    "Avoid the following approaches" block listing one previously-extracted
    approach per round. The reviewer model (vLLM-hosted, OpenAI-compatible)
    extracts a short freeform phrase list from each round's reply; the next
    constraint is sampled per-item with a per-item RNG seed for
    reproducibility without coupling across items.
    """

    rounds: int = 5
    # Same backend types as the judges (vllm/api/azure). In practice we
    # reuse the Qwen-32B-AWQ vLLM that the judges already spin up.
    reviewer_backend: Union[VllmJudgeBackend, ApiJudgeBackend, AzureApiJudgeBackend] = field(
        default_factory=lambda: VllmJudgeBackend(
            base_url="http://localhost:8001/v1",
            model="Qwen/Qwen2.5-32B-Instruct-AWQ",
        )
    )
    reviewer_aut_prompt_id: str = "aut_reviewer_approaches_v1"
    reviewer_macgyver_prompt_id: str = "macgyver_reviewer_approaches_v1"
    reviewer_temperature: float = 0.0
    reviewer_max_tokens: int = 256
    reviewer_max_concurrent: int = 64
    reviewer_max_retries: int = 3
    reviewer_timeout_s: float = 120.0
    reviewer_seed: int = 42
    # Denial-augmented inference prompt ids. Round 0 uses the inference
    # prompt resolved from aut.prompts.inference / cfg.macgyver.prompt_style;
    # rounds 1+ use these.
    aut_denial_prompt_id: str = "aut_user_a_paper_infer_denial_v1"
    macgyver_denial_prompt_id: str = "macgyver_user_default_denial_v1"


# ---------------------------------------------------------------------------
# Brainstorm-then-Select (Summers-Stay 2022) — Phase 2 is its own stage,
# Phase 1 is just aut_infer with temperature>0 + num_inferences>1.
# ---------------------------------------------------------------------------


@dataclass
class ThresholdStrategy:
    """Keep uses whose log-prob confidence exceeds an absolute threshold."""
    type: Literal["threshold"] = "threshold"
    utility_threshold: Optional[float] = None
    originality_threshold: Optional[float] = None
    combine: Literal["and", "or"] = "and"        # only meaningful when both set


@dataclass
class TopFractionStrategy:
    """Rank by one axis (with optional tiebreaker), keep top K%."""
    type: Literal["top_fraction"] = "top_fraction"
    keep_fraction: float = 0.25                   # 0 < frac <= 1
    rank_by: Literal["utility", "originality"] = "originality"
    tiebreaker: Literal["utility", "originality", "none"] = "utility"


@dataclass
class TopFractionIntersectionStrategy:
    """Top-K% on each axis separately, keep the intersection."""
    type: Literal["top_fraction_intersection"] = "top_fraction_intersection"
    keep_fraction: float = 0.25


@dataclass
class TopKPerItemStrategy:
    """Per-item Top-K by combined log-prob score (Summers-Stay-style scoring,
    but grouped by ``item_id`` so every object emits exactly ``k_per_item``
    uses — matching the 8-uses/object convention of the rest of the pipeline).

    ``rank_by="sum"`` ranks on ``utility_conf + originality_conf``; the
    "utility" / "originality" options rank on one axis with the other as a
    tiebreaker. If an item has fewer than K candidates after Phase 1, all of
    them are kept (we never fabricate).
    """
    type: Literal["top_k_per_item"] = "top_k_per_item"
    k_per_item: int = 8
    rank_by: Literal["utility", "originality", "sum"] = "sum"
    tiebreaker: Literal["utility", "originality", "none"] = "originality"


SelectStrategy = Union[
    ThresholdStrategy,
    TopFractionStrategy,
    TopFractionIntersectionStrategy,
    TopKPerItemStrategy,
]


@dataclass
class BrainstormSelectConfig:
    strategy: Union[
        ThresholdStrategy,
        TopFractionStrategy,
        TopFractionIntersectionStrategy,
        TopKPerItemStrategy,
    ]
    skip_chain: bool = False    # skip advantages/drawbacks generation;
                                # score utility/originality directly (faster)
    # Maximum new tokens for the (free-text) advantages/drawbacks generations.
    chain_max_new_tokens: int = 128
    # Slug to emit Phase-2 outputs under (and to label this cell in the
    # aggregate). When the only Phase-1 method is ``baseline``, set this to
    # ``brainstorm`` so the row reads naturally in the summary CSV.
    method_slug: Optional[str] = None


# ---------------------------------------------------------------------------
# Inline column configs (NOT separate stages)
# ---------------------------------------------------------------------------


@dataclass
class OriginalityConfig:
    """Drives the standalone ``aut_novelty`` and ``macgyver_novelty`` stages.

    When set, those stages write ``originality_ngram_<N>`` columns to a
    per-method CSV (one row per use for AUT, one per solution for MacGyver).
    Aggregate then averages each column across rows.
    """
    ngram_index: Optional[Path] = None
    n_values: list[int] = field(default_factory=lambda: [4, 5, 6])


@dataclass
class DiversityConfig:
    """Adds per-method diversity columns to the aggregate summary."""
    embedding_model: str = "all-MiniLM-L6-v2"
    force_cpu: bool = True


# === EXPERIMENTAL [r2_extension] START =====================================
# All r2_extension knobs live under cfg.experimental and default to off
# (verbose_stages=False ⇒ no behavioural change for non-experiment runs).
# To remove the r2_extension feature set: delete this block, remove the
# `experimental:` field from RunConfig, and remove the matching `if
# cfg.experimental ...` guards in steer_collect / steer_extract / aut_infer.
@dataclass
class ExperimentalConfig:
    """Opt-in flags used by the r2_extension experiments only.

    All fields default to no-op so omitting the ``experimental:`` block in a
    YAML leaves pipeline behaviour bit-identical.
    """
    # One-line `[stage] prompt=... bridge_source=... ...` prints at the top
    # of steer_collect / steer_extract / aut_infer. Useful when running a
    # sweep of configs and watching the log scroll by.
    verbose_stages: bool = False
# === EXPERIMENTAL [r2_extension] END =======================================


# ---------------------------------------------------------------------------
# Managed vLLM server (started by the pipeline, torn down at the end)
# ---------------------------------------------------------------------------


@dataclass
class VllmServerSpec:
    """Launch parameters for a vLLM OpenAI-compatible server owned by the run.

    ``managed_vllm.managed_vllm(cfg)`` spawns the server right before the first
    judge stage and tears it down after the last. If ``reuse_existing`` is true
    and an OpenAI-compatible server already answers ``GET {base_url}/models``
    at the inferred URL, the pipeline reuses it (no spawn, no shutdown).
    """

    enable: bool = True
    model: str = "Qwen/Qwen2.5-32B-Instruct-AWQ"
    host: str = "0.0.0.0"
    port: int = 8000
    quantization: Optional[str] = "awq"
    tensor_parallel_size: int = 1
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.90
    dtype: Optional[str] = None
    # Any extra raw flags passed verbatim to vllm.entrypoints.openai.api_server.
    extra_args: list[str] = field(default_factory=list)
    startup_timeout_s: float = 900.0
    poll_interval_s: float = 5.0
    terminate_timeout_s: float = 120.0
    # Reuse an already-healthy server at the inferred base_url instead of spawning.
    reuse_existing: bool = True
    # Leave the server running after this pipeline exits, so the next run can
    # reuse it (with ``reuse_existing: true``) and skip the weight-load cost.
    # When done, kill the whole process group (api_server + EngineCore child)
    # with ``kill -- -<pid>`` — the pipeline prints the exact command on exit.
    keep_alive: bool = False
    # Where to write vLLM's stdout/stderr. ``None`` ⇒ ``<run_dir>/vllm_server.log``.
    log_to: Optional[Path] = None


# ---------------------------------------------------------------------------
# Top-level RunConfig
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    run_id: str
    output_root: Path
    model_name: str
    seed: int

    aut: AUTConfig
    steering: SteeringConfig
    decoding_aut: DecodingConfig

    # Olson-style (prompt-induced) creativity steering. When set, the
    # `olson_collect` stage replaces `steer_collect` as the activation source
    # for `steer_extract`. Mutually exclusive with bridge / CAA-style runs
    # in a single run_dir (each uses its own collector + its own run).
    olson: Optional[OlsonConfig] = None

    # Steered-method inference args. ``None`` ⇒ baseline-only inference.
    aut_steering: Optional[SteeringInferenceConfig] = None
    macgyver_steering: Optional[SteeringInferenceConfig] = None

    macgyver_data: Optional[Path] = None
    macgyver: Optional[MacGyverInferenceConfig] = None
    decoding_macgyver: Optional[DecodingConfig] = None

    aut_judge: Optional[JudgeConfig] = None
    macgyver_judge: Optional[JudgeConfig] = None
    ocsai: Optional[OCSAIConfig] = None

    # Optional brainstorm-select stage (Summers-Stay 2022).
    brainstorm_select: Optional[BrainstormSelectConfig] = None
    # Optional denial-prompting stages (Lu et al. 2024). When set, the
    # ``denial_aut_infer`` and ``denial_macgyver_infer`` stages run T+1
    # rounds per item and append per-round CSVs into the existing
    # ``aut_infer``/``macgyver_infer`` stage directories, so ocsai/judge
    # discovery picks them up alongside ``baseline_results.csv``.
    denial: Optional[DenialConfig] = None
    # Inline column configs — None ⇒ column is absent from the per-row CSV.
    originality: Optional[OriginalityConfig] = None
    diversity:   Optional[DiversityConfig]   = None

    # Lifecycle-managed vLLM judge server. ``None`` ⇒ pipeline assumes a server
    # is already running at the judge backend's ``base_url``.
    vllm_server: Optional[VllmServerSpec] = None

    allow_archived_prompts: bool = False

    # === EXPERIMENTAL [r2_extension] START =================================
    # Opt-in experimental knobs (currently only verbose stage prints).
    # ``None`` ⇒ all experimental flags are off (default).
    experimental: Optional[ExperimentalConfig] = None
    # === EXPERIMENTAL [r2_extension] END ===================================

    def run_dir(self) -> Path:
        return self.output_root / self.run_id


# ---------------------------------------------------------------------------
# YAML loading + validation
# ---------------------------------------------------------------------------


_PATH_FIELDS = {"abcd_data", "macgyver_data", "output_root", "vectors", "ngram_index"}


def _coerce(value: Any, expected: Type) -> Any:
    """Coerce a parsed-YAML value into a dataclass field of type ``expected``."""
    if value is None:
        return None
    origin = get_origin(expected)
    args = get_args(expected)

    # Optional[X] / X | None
    if origin is None and args == ():
        if expected is Path:
            return Path(value)
        return value

    if origin is type(None):  # NoneType
        return None

    # Union (incl. Optional)
    if origin is type(Optional[int]).__mro__[0]:  # Python 3.9 fallback; rare
        pass

    # typing.Union / X | Y
    import typing
    if origin is typing.Union or origin is getattr(__import__("types"), "UnionType", None):
        # try non-None first
        non_none = [a for a in args if a is not type(None)]
        last_err: Exception | None = None
        for a in non_none:
            try:
                return _coerce(value, a)
            except Exception as e:
                last_err = e
        if value is None:
            return None
        if last_err is not None:
            raise last_err
        return value

    if origin in (list, tuple):
        inner = args[0] if args else None
        if inner is None:
            return list(value)
        return [_coerce(v, inner) for v in value]

    if origin is dict:
        return dict(value)

    if origin is Literal:
        if value not in args:
            raise ConfigError(f"Value {value!r} not in allowed literals {args}")
        return value

    if expected is Path:
        return Path(value)

    return value


def _from_dict(cls: Type, data: dict, path: str = "") -> Any:
    """Build a dataclass instance from a dict, rejecting unknown keys."""
    if not is_dataclass(cls):
        return _coerce(data, cls)
    if not isinstance(data, dict):
        raise ConfigError(f"{path or cls.__name__}: expected mapping, got {type(data).__name__}")

    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(
            f"{path or cls.__name__}: unknown key(s) {sorted(unknown)}. "
            f"Allowed: {sorted(known)}"
        )

    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        if name not in data:
            continue
        sub_path = f"{path}.{name}" if path else name
        raw = data[name]
        if is_dataclass(f.type) if isinstance(f.type, type) else False:
            kwargs[name] = _from_dict(f.type, raw, path=sub_path)
        else:
            # Resolve string annotations against module globals
            ftype = _resolve_type(cls, f.type)
            if is_dataclass(ftype) if isinstance(ftype, type) else False:
                kwargs[name] = _from_dict(ftype, raw, path=sub_path)
            else:
                # Optional[Dataclass] / list[Dataclass]
                origin = get_origin(ftype)
                args = get_args(ftype)
                if origin in (list, tuple) and args and is_dataclass(args[0]):
                    kwargs[name] = [_from_dict(args[0], v, path=f"{sub_path}[{i}]")
                                    for i, v in enumerate(raw)]
                else:
                    # Union of dataclasses → tagged-union dispatch on the
                    # ``type`` discriminator if every member has one. Falls
                    # back to "first dataclass" for Optional[Dataclass].
                    dc_members = [
                        a for a in args
                        if isinstance(a, type) and is_dataclass(a)
                    ]
                    if dc_members and isinstance(raw, dict):
                        kwargs[name] = _resolve_tagged_union(
                            dc_members, raw, sub_path,
                        )
                    else:
                        kwargs[name] = _coerce(raw, ftype)
    return cls(**kwargs)


def _resolve_tagged_union(
    members: list[Type],
    raw: dict,
    path: str,
) -> Any:
    """Dispatch a dict to one of N dataclass members by the ``type`` discriminator.

    Each member must have a ``type: Literal["..."]`` field whose default is the
    tag string. If the raw dict has no ``type`` key, we fall back to the first
    dataclass member (Optional[Dataclass] / single-dataclass union case).
    """
    if len(members) == 1:
        return _from_dict(members[0], raw, path=path)

    # Build tag → member from each member's Literal default on its "type" field
    by_tag: dict[str, Type] = {}
    for m in members:
        for f in fields(m):
            if f.name != "type":
                continue
            if get_origin(f.type) is Literal or get_origin(_resolve_type(m, f.type)) is Literal:
                for tag in get_args(_resolve_type(m, f.type)):
                    by_tag[tag] = m
            elif f.default is not None:
                by_tag[str(f.default)] = m

    raw_tag = raw.get("type")
    if raw_tag is None:
        raise ConfigError(
            f"{path}: tagged union requires a 'type' field; "
            f"allowed values: {sorted(by_tag)}"
        )
    if raw_tag not in by_tag:
        raise ConfigError(
            f"{path}: unknown 'type' value {raw_tag!r}; "
            f"allowed: {sorted(by_tag)}"
        )
    return _from_dict(by_tag[raw_tag], raw, path=path)


def _resolve_type(cls: Type, ftype: Any) -> Any:
    """Resolve a string forward-ref to the actual type via the class's module."""
    if isinstance(ftype, str):
        import typing
        ns = vars(__import__(cls.__module__, fromlist=["*"]))
        try:
            return eval(ftype, ns, typing.__dict__)
        except Exception:
            return ftype
    return ftype


def load_run_config(path: str | Path) -> RunConfig:
    """Load YAML → :class:`RunConfig`, validate, run cross-field checks."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text("utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"Top-level YAML must be a mapping, got {type(raw).__name__}")

    cfg = _from_dict(RunConfig, raw)
    validate(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Cross-field validation
# ---------------------------------------------------------------------------


def resolve_aut_prompt(cfg: RunConfig, role: Literal["steering", "inference"]):
    """Return the resolved :class:`PromptSpec` for a given AUT role.

    Order of precedence:
      1. Per-role override ``prompt_id`` (pin by id)
      2. Per-role override ``subject``
      3. ``cfg.aut.prompts.default_subject``
    """
    from .prompts import resolve as resolve_id
    from .prompts.aut import spec_for_subject

    aut_prompts = cfg.aut.prompts
    override: Optional[AUTPromptOverride] = getattr(aut_prompts, role)
    if override is not None:
        if override.prompt_id and override.subject:
            raise ConfigError(
                f"aut.prompts.{role}: set EITHER prompt_id OR subject, not both"
            )
        if override.prompt_id:
            return resolve_id(override.prompt_id,
                              allow_archived=cfg.allow_archived_prompts)
        if override.subject:
            return spec_for_subject(override.subject)
    return spec_for_subject(aut_prompts.default_subject)


def aut_prompt_pair(cfg: RunConfig):
    """Return (steering_spec, inference_spec, decoupled: bool)."""
    s = resolve_aut_prompt(cfg, "steering")
    i = resolve_aut_prompt(cfg, "inference")
    decoupled = (s.id != i.id) or (s.sha256() != i.sha256())
    return s, i, decoupled


def validate(cfg: RunConfig) -> None:
    """Cross-field rules; raises :class:`ConfigError`."""
    from .prompts import resolve as resolve_id

    # --- MacGyver coherence ---------------------------------------------
    if cfg.macgyver_data is not None and cfg.decoding_macgyver is None:
        raise ConfigError(
            "macgyver_data is set but decoding_macgyver is missing — "
            "MacGyver branch requires its own DecodingConfig"
        )
    if cfg.macgyver_judge is not None and cfg.macgyver_data is None:
        raise ConfigError(
            "macgyver_judge is set but macgyver_data is None — "
            "cannot judge a stage that won't run"
        )

    # --- Judge prompt resolution + backend ------------------------------
    for role, jc in (("aut_judge", cfg.aut_judge),
                     ("macgyver_judge", cfg.macgyver_judge)):
        if jc is None:
            continue
        if not jc.prompt_id:
            raise ConfigError(f"{role}.prompt_id is required when {role} is set")
        resolve_id(jc.prompt_id, allow_archived=cfg.allow_archived_prompts)
        _validate_judge_backend(role, jc.backend)

    _validate_aut_decoupling(cfg)
    if cfg.brainstorm_select is not None:
        _validate_brainstorm_select(cfg.brainstorm_select)


def _validate_brainstorm_select(bs: "BrainstormSelectConfig") -> None:
    s = bs.strategy
    if isinstance(s, ThresholdStrategy):
        if s.utility_threshold is None and s.originality_threshold is None:
            raise ConfigError(
                "brainstorm_select.strategy (threshold): set at least one of "
                "utility_threshold / originality_threshold."
            )
    elif isinstance(s, (TopFractionStrategy, TopFractionIntersectionStrategy)):
        if not (0.0 < s.keep_fraction <= 1.0):
            raise ConfigError(
                f"brainstorm_select.strategy.keep_fraction must be in (0, 1], "
                f"got {s.keep_fraction}"
            )
        if isinstance(s, TopFractionStrategy) and s.tiebreaker == s.rank_by:
            raise ConfigError(
                f"brainstorm_select.strategy.tiebreaker ({s.tiebreaker!r}) cannot "
                f"equal rank_by ({s.rank_by!r}); use 'none' to skip a tiebreaker."
            )
    elif isinstance(s, TopKPerItemStrategy):
        if s.k_per_item < 1:
            raise ConfigError(
                f"brainstorm_select.strategy.k_per_item must be >= 1, "
                f"got {s.k_per_item}"
            )
        if s.tiebreaker != "none" and s.tiebreaker == s.rank_by:
            raise ConfigError(
                f"brainstorm_select.strategy.tiebreaker ({s.tiebreaker!r}) cannot "
                f"equal rank_by ({s.rank_by!r}); use 'none' to skip a tiebreaker."
            )
    else:
        raise ConfigError(f"Unknown brainstorm-select strategy: {s!r}")


def _validate_judge_backend(role: str, backend: Any) -> None:
    import os
    if isinstance(backend, ApiJudgeBackend):
        if not backend.api_key_env:
            raise ConfigError(
                f"{role}.backend.api_key_env must name an env var "
                "(the secret itself never goes in the YAML)"
            )
        if not os.environ.get(backend.api_key_env):
            raise ConfigError(
                f"{role}.backend: env var {backend.api_key_env!r} is not set; "
                "export it before running."
            )
    elif isinstance(backend, AzureApiJudgeBackend):
        for attr in ("api_key_env", "endpoint_env", "api_version_env"):
            var_name = getattr(backend, attr)
            if not var_name:
                raise ConfigError(
                    f"{role}.backend.{attr} must name an env var "
                    "(the secret/endpoint itself never goes in the YAML)"
                )
            if not os.environ.get(var_name):
                raise ConfigError(
                    f"{role}.backend: env var {var_name!r} is not set; "
                    "export it before running."
                )
    elif isinstance(backend, VllmJudgeBackend):
        if not backend.base_url:
            raise ConfigError(f"{role}.backend.base_url must be set for vllm backend")


def _validate_aut_decoupling(cfg: "RunConfig") -> None:
    """Resolves both AUT prompt roles, then checks id+hash equality.

    Same-subject prompt version swaps are correctly detected as decoupling.

    Skipped when ``cfg.olson`` is set: Olson runs use ``cfg.olson.*`` for
    activation collection, so ``aut.prompts.steering`` is never consumed and
    decoupling between it and ``aut.prompts.inference`` is meaningless. The
    Olson YAML may set ``aut.prompts.steering = None`` (or omit it) without
    needing ``allow_decoupling: true``.
    """
    if cfg.olson is not None:
        return
    _, _, decoupled = aut_prompt_pair(cfg)
    ap = cfg.aut.prompts
    if decoupled and not ap.allow_decoupling:
        raise ConfigError(
            "AUT prompt decoupling detected (steering vs. inference resolve to "
            "different PromptSpecs). Set aut.prompts.allow_decoupling: true and "
            "give a non-empty decoupling_reason if this is intentional."
        )
    if ap.allow_decoupling and not (ap.decoupling_reason or "").strip():
        raise ConfigError(
            "aut.prompts.allow_decoupling: true requires a non-empty "
            "decoupling_reason"
        )


# ---------------------------------------------------------------------------
# Serialization (for the frozen _config_resolved.yaml snapshot)
# ---------------------------------------------------------------------------


def to_dict(obj: Any) -> Any:
    """Recursive dataclass → dict for JSON / YAML emission. Paths become str."""
    if is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    return obj


def resolved_yaml(cfg: RunConfig) -> str:
    """Serialize cfg back to YAML for ``_config_resolved.yaml``."""
    import yaml

    return yaml.safe_dump(to_dict(cfg), sort_keys=True, default_flow_style=False)
