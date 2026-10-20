"""Astra via the installed Codex CLI, with structured output and bounded calls.

Subscription usage is not metered API cost. max_tokens is advisory for this CLI;
workspace action, optional wall-clock and output-size limits are independently enforced.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from forge_mind.llm.client import LLMError
from forge_mind.llm.codex_diagnostics import CallDiagnostics
from forge_mind.llm.cost import LLMUsage
from forge_mind.llm.output_contract import OutputContractError
from forge_mind.workspace.budget import end_time, expired

T = TypeVar("T", bound=BaseModel)


def _command() -> list[str]:
    explicit = os.environ.get("FORGE_CODEX_EXE")
    if not explicit and os.name == "nt":
        bundled = Path(os.environ.get("LOCALAPPDATA", "")) / "OpenAI/Codex/bin"
        candidates = sorted(
            bundled.glob("*/codex.exe"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if candidates:
            return [str(candidates[0])]
    exe = explicit or shutil.which("codex") or shutil.which("codex.cmd")
    if not exe:
        raise LLMError("Codex CLI is not installed; no model/provider fallback was attempted")
    entry = Path(exe).parent / "node_modules/@openai/codex/bin/codex.js"
    node = shutil.which("node")
    if entry.is_file() and node:
        return [node, str(entry)]
    return [exe]


def _map_schema(schema: dict[str, Any]) -> tuple[dict[str, Any], str | None] | None:
    """Recognize homogeneous maps without conflating them with fixed object fields."""
    if schema.get("type") != "object" or schema.get("properties"):
        return None
    patterns = schema.get("patternProperties")
    additional = schema.get("additionalProperties")
    if patterns:
        if not isinstance(patterns, dict) or len(patterns) != 1:
            raise ValueError("structured map output requires one homogeneous key pattern")
        pattern, item_schema = next(iter(patterns.items()))
        if not isinstance(item_schema, dict):
            raise ValueError("structured map output requires a typed value schema")
        if additional is None or additional is False:
            return item_schema, pattern
        if additional == item_schema:
            return item_schema, None
        raise ValueError(
            "structured map output cannot combine different pattern and fallback values"
        )
    if isinstance(additional, dict):
        return additional, None
    return None


def _resolved_schema(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    """Resolve local references when choosing a wire representation or union branch."""
    visited: set[str] = set()
    while "$ref" in schema:
        reference = schema["$ref"]
        if not isinstance(reference, str) or not reference.startswith("#/"):
            raise ValueError("structured output requires local schema references")
        if reference in visited:
            raise ValueError("cyclic schema reference without a concrete value schema")
        visited.add(reference)
        target = root
        for key in reference.removeprefix("#/").split("/"):
            target = target[key.replace("~1", "/").replace("~0", "~")]
        schema = target
    return schema


def _require_exclusive_tags(schema: dict[str, Any], root: dict[str, Any]) -> None:
    """Lower oneOf only when required wire tags make its branches disjoint."""
    if "anyOf" in schema:
        raise ValueError("structured exclusive unions cannot also contain anyOf")
    variants = [_resolved_schema(candidate, root) for candidate in schema["oneOf"]]
    if not variants or any(variant.get("type") != "object" for variant in variants):
        raise ValueError("structured exclusive unions require disjoint object tags")
    names = set.intersection(*(set(variant.get("properties", {})) for variant in variants))
    discriminator = schema.get("discriminator", {})
    if discriminator:
        name = discriminator.get("propertyName")
        names = {name} if isinstance(name, str) and name in names else set()
    for name in sorted(names):
        seen: set[tuple[str, Any]] = set()
        for variant in variants:
            tag_schema = _resolved_schema(variant["properties"][name], root)
            tags = [tag_schema["const"]] if "const" in tag_schema else tag_schema.get("enum")
            if (
                not isinstance(tags, list)
                or not tags
                or any(type(tag) not in (str, int, float, bool, type(None)) for tag in tags)
            ):
                break
            identities = {
                ("number" if type(tag) in (int, float) else type(tag).__name__, tag) for tag in tags
            }
            if seen.intersection(identities):
                break
            seen.update(identities)
            mapping = discriminator.get("mapping")
            if mapping is not None:
                for tag in tags:
                    reference = mapping.get(str(tag))
                    if (
                        not isinstance(reference, str)
                        or _resolved_schema({"$ref": reference}, root) != variant
                    ):
                        raise ValueError("structured union discriminator mapping is inconsistent")
        else:
            # _strict_schema makes every fixed field required, including tags
            # that have defaults in the original Pydantic model.
            return
    raise ValueError("structured exclusive unions require disjoint object tags")


def _strict_schema(value: Any, root: dict[str, Any] | None = None) -> Any:
    if root is None:
        root = value if isinstance(value, dict) else {}
    if isinstance(value, list):
        return [_strict_schema(v, root) for v in value]
    if not isinstance(value, dict):
        return value
    if "oneOf" in value:
        _require_exclusive_tags(value, root)
    elif "discriminator" in value:
        raise ValueError("structured discriminator requires a proven exclusive union")
    # Length and numeric constraints can stall constrained output generation.
    # Keep them as guidance here; the ORIGINAL Pydantic model still validates
    # every decoded response before any authoring action is admitted.
    local_bounds = {
        k: value[k]
        for k in (
            "minLength",
            "maxLength",
            "minItems",
            "maxItems",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
        )
        if k in value and isinstance(value[k], (int, float)) and not isinstance(value[k], bool)
    }
    out: dict[str, Any] = {}
    for key, child in value.items():
        if key in {"default", "discriminator", *local_bounds}:
            continue
        if key == "oneOf":
            out["anyOf"] = _strict_schema(child, root)
            continue
        if key in {
            "properties",
            "$defs",
            "definitions",
            "patternProperties",
            "dependentSchemas",
        } and isinstance(child, dict):
            # These dictionaries hold NAMES, not schema keywords. Motion
            # drivers have real fields named minimum and maximum.
            out[key] = {name: _strict_schema(schema, root) for name, schema in child.items()}
        else:
            out[key] = _strict_schema(child, root)
    if local_bounds:
        out["description"] = (
            str(out.get("description", ""))
            + " Local validation requires: "
            + json.dumps(local_bounds, sort_keys=True)
        ).strip()
    if "prefixItems" in out:
        items = out.pop("prefixItems")
        out["items"] = items[0] if all(i == items[0] for i in items) else {"anyOf": items}
    mapping = _map_schema(out)
    if mapping is not None:
        item_schema, pattern = mapping
        key_schema = {"type": "string", **({"pattern": pattern} if pattern is not None else {})}
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"key": key_schema, "value": item_schema},
                "required": ["key", "value"],
                "additionalProperties": False,
            },
        }
    if out.get("type") == "object" and "properties" in out:
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False
    return out


def _schema_shape_matches(value: Any, schema: dict[str, Any], root: dict[str, Any]) -> bool:
    """Check branch structure only; the original Pydantic model retains validation."""
    schema = _resolved_schema(schema, root)
    literals = [schema["const"]] if "const" in schema else schema.get("enum")
    if literals is not None and not any(
        value == literal and isinstance(value, bool) == isinstance(literal, bool)
        for literal in literals
    ):
        return False
    kinds = {
        "null": value is None,
        "boolean": isinstance(value, bool),
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": type(value) is int or (type(value) is float and value.is_integer()),
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
    }
    if schema.get("type") in kinds and not kinds[schema["type"]]:
        return False
    variants = schema.get("anyOf", schema.get("oneOf"))
    if variants is not None:
        return any(_schema_shape_matches(value, candidate, root) for candidate in variants)
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", {})
        if not set(schema.get("required", [])).issubset(value):
            return False
        mapping = _map_schema(schema)
        # Fixed object fields are closed in _strict_schema, even when the local
        # model would ignore extras. They cannot identify another wire variant.
        if (
            additional is False or ("properties" in schema and mapping is None)
        ) and value.keys() - properties.keys():
            return False
        return all(
            _schema_shape_matches(
                item,
                mapping[0]
                if mapping
                else properties.get(key, additional if isinstance(additional, dict) else {}),
                root,
            )
            for key, item in value.items()
        )
    if isinstance(value, list):
        prefix = schema.get("prefixItems", [])
        return all(
            _schema_shape_matches(
                item, prefix[i] if i < len(prefix) else schema.get("items", {}), root
            )
            for i, item in enumerate(value)
        )
    return True


def _restore_union(value: Any, schema: dict[str, Any], root: dict[str, Any]) -> Any:
    variants = schema.get("anyOf", schema.get("oneOf", []))
    discriminator = schema.get("discriminator", {})
    if discriminator:
        name = discriminator["propertyName"]
        if not isinstance(value, dict) or name not in value:
            raise ValueError("structured union discriminator is missing")
        mapping = discriminator.get("mapping")
        if mapping is not None:
            reference = mapping.get(str(value[name]))
            if reference is None:
                raise ValueError("structured union discriminator is unknown")
            target = _resolved_schema({"$ref": reference}, root)
            variants = [
                candidate for candidate in variants if _resolved_schema(candidate, root) == target
            ]
    outcomes: dict[str, Any] = {}
    errors: list[ValueError | KeyError] = []
    for candidate in variants:
        try:
            restored = _restore_maps(value, candidate, root)
        except (ValueError, KeyError) as exc:
            errors.append(exc)
            continue
        if _schema_shape_matches(restored, candidate, root):
            outcomes[json.dumps(restored, sort_keys=True)] = restored
    if len(outcomes) > 1:
        raise ValueError("ambiguous structured union map restoration")
    if outcomes:
        return next(iter(outcomes.values()))
    if errors:
        raise errors[0]
    if discriminator:
        raise ValueError("structured union does not match its discriminator")
    # No supported structural conversion exists. Local validators may still
    # accept the unchanged value through coercion or a custom model validator.
    return value


def _restore_maps(value: Any, schema: dict[str, Any], root: dict[str, Any]) -> Any:
    schema = _resolved_schema(schema, root)
    if "anyOf" in schema or "oneOf" in schema:
        return _restore_union(value, schema, root)
    if schema.get("type") == "object":
        mapping = _map_schema(schema)
        if mapping is not None:
            item_schema, _pattern = mapping
            if isinstance(value, list):
                result: dict[str, Any] = {}
                for item in value:
                    if not isinstance(item, dict):
                        raise ValueError("structured map entries require one string key and value")
                    key, mapped_value = item["key"], item["value"]
                    if set(item) != {"key", "value"} or not isinstance(key, str):
                        raise ValueError("structured map entries require one string key and value")
                    if key in result:
                        raise ValueError("duplicate map keys in structured output")
                    result[key] = _restore_maps(mapped_value, item_schema, root)
                return result
            if isinstance(value, dict):
                return {key: _restore_maps(item, item_schema, root) for key, item in value.items()}
        if isinstance(value, dict):
            return {
                k: _restore_maps(v, schema.get("properties", {}).get(k, {}), root)
                for k, v in value.items()
            }
    if schema.get("type") == "array" and isinstance(value, list):
        return [_restore_maps(v, schema.get("items", {}), root) for v in value]
    return value


class CodexClient:
    def __init__(
        self, *, timeout_s: float | None = 300, cancelled: Callable[[], bool] | None = None
    ) -> None:
        self.timeout_s = timeout_s
        self.cancelled = cancelled or (lambda: False)
        self.model = os.environ.get("FORGE_CODEX_MODEL", "gpt-6-astra")
        self._last_input_images: tuple[dict[str, Any], ...] = ()
        self._last_call_diagnostics: dict[str, Any] | None = None

    @property
    def last_input_images(self) -> tuple[dict[str, Any], ...]:
        """Images attached to the last launched CLI request, not proof of model receipt."""
        return tuple(dict(item) for item in self._last_input_images)

    @property
    def last_call_diagnostics(self) -> dict[str, Any] | None:
        """Sanitized evidence for the latest attempt, including unsuccessful calls."""
        return deepcopy(self._last_call_diagnostics)

    def _generate(
        self,
        system: str,
        user: str,
        output_model: type[T],
        max_tokens: int,
        image_png: bytes | None = None,
        *,
        live_web: bool = False,
        images_png: Sequence[bytes] | None = None,
        image_labels: Sequence[str] | None = None,
    ) -> tuple[T, LLMUsage]:
        diagnostics = CallDiagnostics(self.model)
        self._last_call_diagnostics = None
        try:
            return self._generate_attempt(
                system,
                user,
                output_model,
                max_tokens,
                image_png,
                diagnostics=diagnostics,
                live_web=live_web,
                images_png=images_png,
                image_labels=image_labels,
            )
        except BaseException as exc:
            if diagnostics.collected and diagnostics.primary_exception is not exc:
                diagnostics.record["cleanup_error"] = True
            if isinstance(exc, OutputContractError):
                error = exc._with_retryable(
                    diagnostics.output_correction_allowed()
                    and not self.cancelled()
                    and (
                        self.timeout_s is None
                        or time.monotonic() < diagnostics.started + self.timeout_s
                    )
                )
                diagnostics.rejected_output(error)
                raise error from exc.__cause__
            raise
        finally:
            self._last_call_diagnostics = diagnostics.finish()

    def _generate_attempt(
        self,
        system: str,
        user: str,
        output_model: type[T],
        max_tokens: int,
        image_png: bytes | None = None,
        *,
        diagnostics: CallDiagnostics,
        live_web: bool = False,
        images_png: Sequence[bytes] | None = None,
        image_labels: Sequence[str] | None = None,
    ) -> tuple[T, LLMUsage]:
        self._last_input_images = ()
        deadline = end_time(self.timeout_s)
        cancelled = self.cancelled()
        if cancelled or expired(deadline):
            diagnostics.record["outcome"] = "cancelled" if cancelled else "timeout"
            raise LLMError("Codex call cancelled or exceeded its wall-clock limit")
        if images_png is not None:
            images = tuple(images_png)
            labels = tuple(image_labels or ())
            if image_png is not None or not images or len(images) != len(labels):
                raise LLMError("Codex image request requires one label for every ordered image")
            if any(not isinstance(value, bytes) or not value for value in images):
                raise LLMError("Codex image bytes must all be available before invocation")
            if any(not isinstance(label, str) or not label.strip() for label in labels):
                raise LLMError("Codex image labels must be nonempty strings")
        else:
            images = (image_png,) if image_png is not None else ()
            labels = ("Primary image",) if images else ()
        image_inputs = tuple(
            {
                "index": index,
                "label": label,
                "sha256": hashlib.sha256(value).hexdigest(),
                "bytes": len(value),
            }
            for index, (label, value) in enumerate(zip(labels, images, strict=True), start=1)
        )
        diagnostics.record["outcome"] = "setup_failure"
        with (
            tempfile.TemporaryDirectory(prefix="forge-codex-") as tmp,
            diagnostics.scope(Path(tmp)),
        ):
            root = Path(tmp)
            schema = root / "schema.json"
            original_schema = output_model.model_json_schema()
            schema.write_text(json.dumps(_strict_schema(original_schema)), encoding="utf-8")
            output = root / "answer.json"
            cmd = [
                *_command(),
                "exec",
                "--ignore-user-config",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--model",
                self.model,
                "-c",
                "features.shell_tool=false",
                "-c",
                'web_search="live"' if live_web else 'web_search="disabled"',
                "--output-schema",
                str(schema),
                "--output-last-message",
                str(output),
                "--json",
                "--color",
                "never",
            ]
            for index, value in enumerate(images):
                attached = root / ("input.png" if index == 0 else f"input-{index + 1}.png")
                attached.write_bytes(value)
                cmd.extend(["--image", str(attached)])
            cmd.append("-")
            env = {
                k: v
                for k, v in os.environ.items()
                if k not in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"}
            }
            prompt = (
                "Return only the requested JSON. Use web search only; do not inspect files.\n"
                if live_web
                else "Return only the requested JSON. Do not call tools or inspect files.\n"
            ) + f"Answer budget approximately {max_tokens} tokens.\n{system}\n\n{user}"
            if images_png is not None:
                prompt = (
                    "Attached image order and scope (display context, not validation evidence):\n"
                    + json.dumps(image_inputs, ensure_ascii=True)
                    + "\n\n"
                    + prompt
                )
            with (
                (root / "events.jsonl").open("w+b") as events,
                (root / "stderr.txt").open("w+b") as stderr,
            ):
                diagnostics.record["outcome"] = "spawn_failure"
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=events,
                    stderr=stderr,
                    cwd=root,
                    env=env,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                diagnostics.record["process_started"] = True
                diagnostics.record["outcome"] = "internal_error"
                self._last_input_images = image_inputs
                assert proc.stdin is not None

                def feed() -> None:
                    try:
                        assert proc.stdin is not None
                        proc.stdin.write(prompt.encode("utf-8"))
                        proc.stdin.close()
                    except (OSError, ValueError):
                        pass  # The parent reports the child exit or cancellation.

                writer = threading.Thread(target=feed, daemon=True)
                writer.start()
                from forge_mind.agentic.sandbox import _kill_tree

                try:
                    while proc.poll() is None:
                        cancelled = self.cancelled()
                        if cancelled or expired(deadline):
                            diagnostics.record["outcome"] = "cancelled" if cancelled else "timeout"
                            raise LLMError("Codex call cancelled or exceeded its wall-clock limit")
                        if any(
                            (root / n).stat().st_size > 4 * 1024 * 1024
                            for n in ("events.jsonl", "stderr.txt")
                        ):
                            diagnostics.record["outcome"] = "output_overflow"
                            raise LLMError("Codex logs exceeded their output-byte limit")
                        time.sleep(0.1)
                finally:
                    try:
                        if proc.poll() is None:
                            _kill_tree(proc.pid)
                        proc.wait(timeout=10)
                        writer.join(timeout=2)
                    except BaseException:
                        diagnostics.record["cleanup_error"] = True
                        raise
                    finally:
                        diagnostics.record["returncode"] = proc.returncode
                if proc.returncode != 0 or not output.exists():
                    diagnostics.record["outcome"] = (
                        "nonzero_exit" if proc.returncode != 0 else "invalid_response"
                    )
                    if proc.returncode == 0:
                        raise diagnostics.rejected_output(
                            OutputContractError(stage="missing_response", schema=original_schema)
                        )
                    events.seek(0)
                    reason = "check CLI authentication/model access"
                    for line in events:
                        try:
                            event = json.loads(line)
                            if event.get("type") == "turn.failed":
                                reason = str(event.get("error", {}).get("message", reason))[:1200]
                        except (ValueError, AttributeError):
                            continue
                    raise LLMError(
                        f"Codex structured call failed (exit={proc.returncode}): {reason}"
                    )
                if output.stat().st_size > 512_000:
                    diagnostics.record["outcome"] = "output_overflow"
                    diagnostics.record["output_contract"] = {
                        **OutputContractError(
                            stage="output_overflow", schema=original_schema
                        ).feedback,
                        "retryable": False,
                    }
                    raise LLMError("Codex response exceeded 512 KB")
                diagnostics.record["outcome"] = "invalid_response"
                try:
                    raw = output.read_text(encoding="utf-8")
                except UnicodeDecodeError as exc:
                    raise diagnostics.rejected_output(
                        OutputContractError(
                            stage="utf8_decode",
                            schema=original_schema,
                            violations=[{"type": "utf8_invalid"}],
                        )
                    ) from exc
                try:
                    decoded = json.loads(raw)
                except ValueError as exc:
                    raise diagnostics.rejected_output(
                        OutputContractError(
                            stage="json_decode",
                            schema=original_schema,
                            violations=[{"type": "json_invalid"}],
                        )
                    ) from exc
                try:
                    restored = _restore_maps(decoded, original_schema, original_schema)
                except (ValueError, KeyError) as exc:
                    error = OutputContractError(
                        stage="map_restore",
                        schema=original_schema,
                        violations=[{"type": "map_restore_invalid"}],
                    )
                    diagnostics.record["output_contract"] = {**error.feedback, "retryable": False}
                    if isinstance(exc, KeyError):
                        raise
                    raise error from exc
                try:
                    parsed = output_model.model_validate(restored)
                except ValidationError as exc:
                    raise diagnostics.rejected_output(
                        OutputContractError(
                            stage="pydantic_validation",
                            schema=original_schema,
                            violations=exc.errors(
                                include_url=False, include_context=False, include_input=False
                            ),
                        )
                    ) from exc
                diagnostics.record["outcome"] = "internal_error"
                events.seek(0)
                usage: dict[str, int] = {}
                for line in events:
                    try:
                        item = json.loads(line)
                        if item.get("type") == "turn.completed":
                            usage = item.get("usage", {})
                    except (ValueError, AttributeError):
                        continue
                result_usage = LLMUsage(
                    model=self.model,
                    input_tokens=max(
                        0, usage.get("input_tokens", 0) - usage.get("cached_input_tokens", 0)
                    ),
                    output_tokens=usage.get("output_tokens", 0),
                    cache_read_tokens=usage.get("cached_input_tokens", 0),
                    cost_usd=0.0,
                )
                diagnostics.record["outcome"] = "success"
                return parsed, result_usage

    def research_structured(
        self, *, system: str, user: str, output_model: type[T], max_tokens: int = 3000
    ) -> tuple[T, LLMUsage]:
        """Explicit subscription-only research; ordinary author calls keep web disabled."""
        return self._generate(system, user, output_model, max_tokens, live_web=True)

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        output_model: type[T],
        model: str,
        max_tokens: int,
        cache_system: bool = True,
    ) -> tuple[T, LLMUsage]:
        return self._generate(system, user, output_model, max_tokens)

    def generate_structured_with_image(
        self,
        *,
        system: str,
        user: str,
        image_png: bytes,
        output_model: type[T],
        model: str,
        max_tokens: int,
    ) -> tuple[T, LLMUsage]:
        return self._generate(system, user, output_model, max_tokens, image_png)

    def generate_structured_with_images(
        self,
        *,
        system: str,
        user: str,
        images_png: Sequence[bytes],
        image_labels: Sequence[str],
        output_model: type[T],
        model: str,
        max_tokens: int,
    ) -> tuple[T, LLMUsage]:
        """Attach ordered, labeled images in one request; no fallback or automatic retry.

        Callers keep full context first and decide whether a supplemental inspection
        image is valid. This transport does not infer roles, revision identity or validity.
        """
        if images_png is None:
            self._last_call_diagnostics = CallDiagnostics(self.model).finish()
            self._last_input_images = ()
            raise LLMError("Codex image bytes must all be available before invocation")
        return self._generate(
            system,
            user,
            output_model,
            max_tokens,
            images_png=images_png,
            image_labels=image_labels,
        )
