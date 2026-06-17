import asyncio
import json
import random
import re
from contextlib import nullcontext
from typing import Any

import numpy as np
import pandas as pd
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.exceptions import UnexpectedModelBehavior
from rich.console import Console
from rich import print
from rich.traceback import install as install_rich_traceback

from tabautosyn.utils.langfuse import (
    get_langfuse_judge_client,
    langfuse_output_payload,
    langfuse_safe_end,
    langfuse_safe_trace,
    langfuse_safe_update,
)

from tabautosyn.agents.prompts import (
    DEPENDENT_RANGE_BATCH_DETECTOR_FORMAT_REMINDER,
    DEPENDENT_RANGE_BATCH_DETECTOR_PROMPT,
    DEPENDENT_RANGE_BATCH_DETECTOR_USER_PROMPT,
    ENCODING_CHECKER_PROMPT,
)

install_rich_traceback(show_locals=False)
RICH_CONSOLE = Console()


def _emit_langfuse_batch_validation_summary(summary: dict[str, Any]) -> None:
    """Single Langfuse span for high-frequency DependencyFixer batch validation (aggregates only).

    Per-batch ``DependencyViolationDetectorAgent`` runs stay ``instrument=False`` to avoid
    flooding Langfuse; EncodingChecker uses a separate root trace per run when ``langfuse_client`` is set.
    """
    try:
        lf = get_langfuse_judge_client()
        if not hasattr(lf, "start_span"):
            return
        span = lf.start_span(
            name="DependencyFixer.batch_validation_summary",
            input={
                "component": "DependencyFixer.llm_refine_dependent_ranges",
                "note": (
                    "Aggregated counts for batched dependency-violation LLM checks only. "
                    "EncodingChecker traces separately; per-batch detector spans omitted here."
                ),
            },
            metadata=summary,
        )
        span.update(output=summary)
        span.end()
    except Exception:
        return


class DependencyFixer:
    """Post-processes a synthetic DataFrame to enforce column dependencies
    discovered from the real data.

    Each dependency type has a dedicated ``_fix_*`` method that either
    corrects or drops rows in the synthetic DataFrame so that the
    statistical relationships observed in the real data are preserved.

    ``dependent_range`` with fewer than 3 columns in the payload are handled
    by :meth:`_fix_dependent_range` inside :meth:`fix_dependencies`. Wider
    ranges are refined with an LLM via :meth:`fix_dependencies_async`.
    """

    _MAX_SINGLE_FILTER_DROP_FRACTION = 0.60
    _MIN_FILTERED_ROWS_FRACTION = 0.35
    _ENCODING_CHECKER_MAX_ANCHORS_PER_RUN = 60
    _DUPLICATE_RETRY_TEMPERATURE_START = 0.35
    _DUPLICATE_RETRY_TEMPERATURE_STEP = 0.15
    _DUPLICATE_RETRY_TEMPERATURE_MAX = 0.90

    def __init__(
        self, syn_df: pd.DataFrame, real_df: pd.DataFrame, dependencies: dict[str, list]
    ):

        self.syn_df = syn_df.copy()
        self.real_df = real_df
        self.dependencies = self._filter_dependencies(dependencies)
        self._pending_llm_dependent_ranges: list[dict] = []
        self._had_llm_dependent_range_pass: bool = False
        self._row_filter_baseline_count: int = len(self.syn_df)
        self.verbose: bool = False
        self._fixer_segment_label: str | None = None
        self.llm_usage_summary: dict[str, float | int] = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }
        self.per_agent_usage_summary: dict[str, dict[str, float | int]] = {}

    @property
    def had_llm_dependent_range_pass(self) -> bool:
        """True after :meth:`fix_dependencies_async` if any LLM ``dependent_range`` pass ran."""
        return self._had_llm_dependent_range_pass

    @staticmethod
    def _filter_dependencies(dependencies: dict):
        """Keep dependencies whose confidence is at least 0.7.

        LLM discovery is schema-guided but not schema-guaranteed; skip malformed
        entries instead of treating them as trusted dependencies.
        """
        filtered: dict[str, list[dict]] = {}
        for dep_type, deps in (dependencies or {}).items():
            if not isinstance(deps, list):
                continue

            kept = []
            for dep in deps:
                if not isinstance(dep, dict):
                    continue
                normalized_dep = dep.copy()
                if "confidence" not in normalized_dep:
                    continue
                try:
                    confidence = float(normalized_dep["confidence"])
                except (TypeError, ValueError):
                    continue
                normalized_dep["confidence"] = confidence
                if confidence >= 0.7:
                    kept.append(normalized_dep)

            if kept:
                filtered[dep_type] = kept

        return filtered

    @staticmethod
    def _extract_usage_summary(run_result: Any) -> dict[str, float | int]:
        """Extract token/cost counters from a Pydantic-AI run result."""
        usage_obj = None
        try:
            usage_obj = run_result.usage()
        except Exception:
            return {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }

        def _as_int(value: Any) -> int:
            if value is None:
                return 0
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        def _as_float(value: Any) -> float:
            if value is None:
                return 0.0
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        requests = _as_int(getattr(usage_obj, "requests", 0))
        input_tokens = _as_int(
            getattr(usage_obj, "input_tokens", getattr(usage_obj, "request_tokens", 0))
        )
        output_tokens = _as_int(
            getattr(usage_obj, "output_tokens", getattr(usage_obj, "response_tokens", 0))
        )
        total_tokens_raw = getattr(usage_obj, "total_tokens", None)
        total_tokens = _as_int(
            total_tokens_raw if total_tokens_raw is not None else input_tokens + output_tokens
        )
        cost_usd = _as_float(
            getattr(usage_obj, "cost", getattr(usage_obj, "total_cost", 0.0))
        )

        return {
            "requests": requests,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
        }

    def _accumulate_usage_summary(
        self, run_result: Any, agent_name: str | None = None
    ) -> None:
        usage = self._extract_usage_summary(run_result)
        self.llm_usage_summary["requests"] += int(usage["requests"])
        self.llm_usage_summary["input_tokens"] += int(usage["input_tokens"])
        self.llm_usage_summary["output_tokens"] += int(usage["output_tokens"])
        self.llm_usage_summary["total_tokens"] += int(usage["total_tokens"])
        self.llm_usage_summary["cost_usd"] += float(usage["cost_usd"])
        if agent_name:
            if agent_name not in self.per_agent_usage_summary:
                self.per_agent_usage_summary[agent_name] = {
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                }
            self.per_agent_usage_summary[agent_name]["requests"] += int(
                usage["requests"]
            )
            self.per_agent_usage_summary[agent_name]["input_tokens"] += int(
                usage["input_tokens"]
            )
            self.per_agent_usage_summary[agent_name]["output_tokens"] += int(
                usage["output_tokens"]
            )
            self.per_agent_usage_summary[agent_name]["total_tokens"] += int(
                usage["total_tokens"]
            )
            self.per_agent_usage_summary[agent_name]["cost_usd"] += float(
                usage["cost_usd"]
            )

    def _filter_rows_conservatively(
        self,
        syn_df: pd.DataFrame,
        keep_mask: Any,
        dep_type: str,
        expression: str,
    ) -> pd.DataFrame:
        """Apply a row filter unless it would remove a suspiciously large slice."""
        try:
            before = len(syn_df)
            if before == 0:
                return syn_df

            kept = int(keep_mask.sum())
            dropped = before - kept
            if dropped <= 0:
                return syn_df

            single_drop_fraction = dropped / before
            baseline = max(self._row_filter_baseline_count, 1)
            total_keep_fraction = kept / baseline

            too_aggressive = (
                single_drop_fraction > self._MAX_SINGLE_FILTER_DROP_FRACTION
                or total_keep_fraction < self._MIN_FILTERED_ROWS_FRACTION
            )
            if too_aggressive:
                if self.verbose:
                    print(
                        "[yellow]Skipping aggressive dependency filter[/yellow] "
                        f"({dep_type}: {expression!r}) would drop {dropped}/{before} rows."
                    )
                return syn_df

            return syn_df[keep_mask]
        except Exception as exc:
            if self.verbose:
                print(
                    "[yellow]Skipping invalid dependency filter[/yellow] "
                    f"({dep_type}: {expression!r}): {exc}"
                )
            return syn_df

    def _fix_mapping(
        self, mapping: list[dict], real_df: pd.DataFrame, syn_df: pd.DataFrame
    ):
        """Fix one-to-one mapping dependencies (e.g. ``country -> currency``).

        Builds a lookup from the real data where the anchor column
        deterministically maps to a single value in the target column,
        then applies that mapping to the synthetic data.
        """
        for dep in mapping:
            expression = dep["expression"]
            columns = dep["columns"]
            anchor_column = dep["anchor_column"]
            if not self._check_columns_in_real(columns, real_df):
                continue

            expression = self._strip_spaces(expression, columns)
            first_col, second_col = expression.split("->")

            if first_col != anchor_column:
                continue

            mapping_dict = {}
            for key in real_df[first_col].unique():
                df_col = real_df[real_df[first_col] == key]
                value = df_col[second_col].unique()
                if len(value) == 1:
                    mapping_dict[key] = value[0]

            for idx, row in syn_df.iterrows():
                mapped = mapping_dict.get(row[first_col])
                if mapped is not None:
                    syn_df.at[idx, second_col] = mapped

        return syn_df.reset_index(drop=True)

    def _fix_dependent_range(
        self, dependent_range: list[dict], real_df: pd.DataFrame, syn_df: pd.DataFrame
    ):
        """Drop rows whose dependent columns violate dependent-range constraints.

        When multiple dependencies target the same columns, only the one
        with the highest confidence is applied.

        Preference order for constraints:
        1) explicit ``value_map`` from dependency payload
        2) fallback constraints derived from ``real_df``
        """
        best_by_target: dict[frozenset, dict] = {}
        for dep in dependent_range:
            dep_cols = frozenset(
                col for col in dep["columns"] if col != dep["anchor_column"]
            )
            prev = best_by_target.get(dep_cols)
            if prev is None or float(dep["confidence"]) > float(prev["confidence"]):
                best_by_target[dep_cols] = dep

        for dep in best_by_target.values():
            columns = dep["columns"]
            anchor_column = dep["anchor_column"]

            if not self._check_columns_in_real(columns, real_df):
                continue

            dependent_cols = [col for col in columns if col != anchor_column]

            if not dependent_cols:
                continue

            allowed_values: dict = {}
            keys = real_df[anchor_column].unique()
            value_map = dep.get("value_map", {})
            has_value_map = isinstance(value_map, dict) and len(value_map) > 0

            anchor_dtype = real_df[anchor_column].dtype

            continuous_cols = set(
                col
                for col in dependent_cols
                if pd.api.types.is_numeric_dtype(real_df[col])
            )

            if has_value_map:
                for raw_key, col_map in value_map.items():
                    if not isinstance(col_map, dict):
                        continue
                    try:
                        if pd.api.types.is_integer_dtype(anchor_dtype):
                            coerced_key = int(float(raw_key))
                        elif pd.api.types.is_float_dtype(anchor_dtype):
                            coerced_key = float(raw_key)
                        else:
                            coerced_key = raw_key
                    except (ValueError, TypeError):
                        coerced_key = raw_key

                    key_vals = {}
                    for col in dependent_cols:
                        col_payload = col_map.get(col, {})
                        values = (
                            col_payload.get("values", [])
                            if isinstance(col_payload, dict)
                            else []
                        )
                        value_mode = (
                            col_payload.get("value_mode")
                            if isinstance(col_payload, dict)
                            else None
                        )
                        if not isinstance(values, list):
                            values = []
                        cleaned = [v for v in values if pd.notna(v)]
                        if col in continuous_cols:
                            arr = (
                                np.sort(np.array(cleaned, dtype=float))
                                if cleaned
                                else np.array([])
                            )
                            inferred_mode = value_mode
                            if inferred_mode not in {"set", "range"}:
                                # Backward-compatible inference:
                                # 2 numeric endpoints are commonly emitted as [min, max].
                                inferred_mode = "range" if len(arr) == 2 else "set"
                            key_vals[col] = {
                                "mode": inferred_mode,
                                "values": arr,
                            }
                        else:
                            key_vals[col] = {
                                "mode": "set",
                                "values": set(cleaned),
                            }
                    allowed_values[coerced_key] = key_vals

            # Fallback for missing/incomplete anchors from value_map.
            for key in keys:
                if key in allowed_values and all(
                    col in allowed_values[key] for col in dependent_cols
                ):
                    continue
                df_key = real_df[real_df[anchor_column] == key]
                key_vals = allowed_values.get(key, {})
                for col in dependent_cols:
                    if col in key_vals:
                        continue
                    vals = df_key[col].dropna()
                    if col in continuous_cols:
                        key_vals[col] = {
                            "mode": "range",
                            "values": np.sort(vals.unique()),
                        }
                    else:
                        key_vals[col] = {
                            "mode": "set",
                            "values": set(vals.unique()),
                        }
                allowed_values[key] = key_vals

            mask = pd.Series(True, index=syn_df.index)
            for key in syn_df[anchor_column].unique():
                df_subset = syn_df[syn_df[anchor_column] == key]
                if key not in allowed_values:
                    mask[df_subset.index] = False
                    continue
                for idx, row in df_subset.iterrows():
                    valid = True
                    for col in dependent_cols:
                        val = row[col]
                        if pd.isna(val):
                            valid = False
                            break
                        ref_info = allowed_values[key][col]
                        if col in continuous_cols:
                            ref_values = ref_info.get("values", np.array([]))
                            ref_mode = ref_info.get("mode", "range")
                            if len(ref_values) == 0:
                                valid = False
                                break
                            if ref_mode == "set":
                                # For discrete allowed numeric sets use exact match with tolerance.
                                if not np.isclose(
                                    ref_values, val, rtol=1e-6, atol=1e-8
                                ).any():
                                    valid = False
                                    break
                            else:
                                ref_min, ref_max = ref_values[0], ref_values[-1]
                                if has_value_map:
                                    # Explicit value_map with range mode: strict bounds.
                                    if val < ref_min or val > ref_max:
                                        valid = False
                                        break
                                else:
                                    span = (
                                        ref_max - ref_min
                                        if len(ref_values) > 1
                                        else (
                                            abs(ref_values[0])
                                            if ref_values[0] != 0
                                            else 1.0
                                        )
                                    )
                                    margin = 0.05 * span
                                    if val < ref_min - margin or val > ref_max + margin:
                                        valid = False
                                        break
                        else:
                            ref_values = ref_info.get("values", set())
                            if val not in ref_values:
                                valid = False
                                break
                    if not valid:
                        mask[idx] = False

            syn_df = self._filter_rows_conservatively(
                syn_df,
                mask,
                "dependent_range",
                dep.get("expression", ""),
            )

        return syn_df.reset_index(drop=True)

    def _fix_rule(self, rule: dict, real_df: pd.DataFrame, syn_df: pd.DataFrame):
        """Fix rule-based dependencies (e.g. ``age >= 0``).

        The expression is first validated against the real data. Invalid
        expressions are skipped; valid synthetic violations are filtered
        conservatively.
        """
        for dep in rule:
            columns = dep["columns"]
            expression = dep["expression"]

            if not self._check_columns_in_real(columns, real_df):
                continue

            expression = self._strip_spaces(expression, columns)

            real_expression = self._sub_columns(expression, columns, "real_df")

            try:
                result = eval(real_expression)
                if not all(result):
                    continue
            except Exception as exc:
                if self.verbose:
                    print(
                        "[yellow]Skipping invalid rule dependency[/yellow] "
                        f"({expression!r}): {exc}"
                    )
                continue

            syn_work = syn_df.copy()
            for col in columns:
                if col in syn_work.columns and col in real_df.columns:
                    syn_work[col] = self._align_syn_series_dtype(
                        real_df[col], syn_work[col]
                    )
            syn_expression = self._sub_columns(expression, columns, "syn_work")

            try:
                eval_result = eval(syn_expression, {"syn_work": syn_work})
            except Exception as exc:
                if self.verbose:
                    print(
                        "[yellow]Skipping invalid synthetic rule dependency[/yellow] "
                        f"({expression!r}): {exc}"
                    )
                continue
            syn_df = self._filter_rows_conservatively(
                syn_df,
                eval_result,
                "rule",
                expression,
            )

        return syn_df.reset_index(drop=True)

    def _fix_range(self, range: dict, real_df: pd.DataFrame, syn_df: pd.DataFrame):
        """Clip synthetic values to the [min, max] range observed in the real data.

        Each column listed in the dependency is clipped independently.
        """
        for dep in range:
            columns = dep["columns"]
            if not self._check_columns_in_real(columns, real_df):
                continue

            for col in columns:
                if col not in syn_df.columns:
                    continue
                real_series = real_df[col]
                if not pd.api.types.is_numeric_dtype(real_series):
                    continue
                aligned = self._align_syn_series_dtype(real_series, syn_df[col])
                col_min = real_series.min()
                col_max = real_series.max()
                syn_df[col] = aligned.clip(lower=col_min, upper=col_max)

        return syn_df.reset_index(drop=True)

    def _fix_correspondence(
        self, correspondence: dict, real_df: pd.DataFrame, syn_df: pd.DataFrame
    ):
        """Fix correspondence dependencies (e.g. ``A + B == C``).

        If one side of the equality is a single column and the other is
        a formula, the column is recalculated from the formula.  Otherwise
        rows that violate the expression are dropped.

        Validated against real data first — the rule is skipped when the
        real data itself does not satisfy it.
        """
        for dep in correspondence:
            columns = dep["columns"]
            expression = dep["expression"]
            if not self._check_columns_in_real(columns, real_df):
                continue

            expression = self._strip_spaces(expression, columns)
            expression = re.sub(r"(?<!=)=(?!=)", "==", expression)
            validation_expression = self._rewrite_chained_equalities(expression)

            real_expression = self._sub_columns(
                validation_expression, columns, "real_df"
            )
            try:
                result = eval(real_expression)
            except (SyntaxError, TypeError, NameError, ValueError):
                continue
            try:
                if not all(result):
                    continue
            except (TypeError, ValueError):
                continue

            target_col, formula = self._split_correspondence(expression, columns)

            if target_col is not None:
                formula_expr = self._sub_columns(formula, columns, "syn_df")
                try:
                    syn_df[target_col] = eval(formula_expr)
                except (SyntaxError, TypeError, NameError):
                    continue
            else:
                syn_expression = self._sub_columns(
                    validation_expression, columns, "syn_df"
                )
                try:
                    eval_result = eval(syn_expression)
                except (SyntaxError, TypeError, NameError, ValueError):
                    continue
                syn_df = self._filter_rows_conservatively(
                    syn_df,
                    eval_result,
                    "correspondence",
                    expression,
                )

        return syn_df.reset_index(drop=True)

    @staticmethod
    def _rewrite_chained_equalities(expression: str) -> str:
        """Rewrite chained equalities to element-wise comparisons.

        Python interprets ``a == b == c`` as ``(a == b) and (b == c)``,
        which is invalid for pandas Series because ``and`` requires scalar
        truth values. This converts it to ``(a == b) & (b == c)``.
        """
        parts = [part.strip() for part in expression.split("==")]
        if len(parts) <= 2:
            return expression

        pairwise = [f"(({parts[i]})==({parts[i + 1]}))" for i in range(len(parts) - 1)]
        return "&".join(pairwise)

    @staticmethod
    def _split_correspondence(expression: str, columns: list[str]):
        """Split ``formula == col`` or ``col == formula`` into (col, formula).

        Returns ``(target_column, formula_str)`` when exactly one side
        of ``==`` is a bare column name.  Returns ``(None, None)`` when
        neither side is a single column (e.g. both sides are formulas).
        """
        parts = expression.split("==")
        if len(parts) != 2:
            return None, None

        left, right = parts[0].strip(), parts[1].strip()

        left_is_col = left in columns
        right_is_col = right in columns

        if right_is_col and not left_is_col:
            return right, left
        if left_is_col and not right_is_col:
            return left, right
        return None, None

    def _fix_logic(
        self, logic: list[dict], real_df: pd.DataFrame, syn_df: pd.DataFrame
    ):
        """Fix conditional logic dependencies.

        Handles expressions of the form
        ``[if] <condition> then <consequence>``
        (e.g. ``if hours-per-week >= 40 then income == 1``).

        Rows where the condition holds but the consequence does not are
        filtered conservatively. Invalid expressions are skipped.
        """
        for dep in logic:
            columns = dep["columns"]
            expression = dep["expression"]
            if not self._check_columns_in_real(columns, real_df):
                continue

            expr = expression.strip()
            expr_lower = expr.lower()

            if expr_lower.startswith("if "):
                expr = expr[3:].strip()
                expr_lower = expr_lower[3:].strip()

            then_pos = expr_lower.find(" then ")
            if then_pos == -1:
                continue

            condition_part = expr[:then_pos].strip()
            consequence_part = expr[then_pos + 6 :].strip()

            condition_expr = self._strip_spaces(condition_part, columns)
            consequence_expr = self._strip_spaces(consequence_part, columns)

            real_condition = self._sub_columns(condition_expr, columns, "real_df")
            real_consequence = self._sub_columns(consequence_expr, columns, "real_df")

            try:
                cond_mask_real = eval(real_condition)
                cons_mask_real = eval(real_consequence)
                if not all(cons_mask_real[cond_mask_real]):
                    continue
            except (SyntaxError, TypeError, NameError, ValueError):
                continue

            syn_condition = self._sub_columns(condition_expr, columns, "syn_df")
            syn_consequence = self._sub_columns(consequence_expr, columns, "syn_df")

            try:
                cond_mask = eval(syn_condition)
                cons_mask = eval(syn_consequence)
                violating = cond_mask & ~cons_mask
            except (SyntaxError, TypeError, NameError, ValueError):
                continue
            if not violating.any():
                continue

            syn_df = self._filter_rows_conservatively(
                syn_df,
                ~violating,
                "logic",
                expression,
            )

        return syn_df.reset_index(drop=True)

    def _fix_temporal_ordering(
        self, temporal_ordering: dict, real_df: pd.DataFrame, syn_df: pd.DataFrame
    ):
        """Fix temporal ordering dependencies (e.g. ``start_date <= end_date``).

        Validated against real data first; synthetic rows that violate
        the ordering expression are dropped.
        """
        for dep in temporal_ordering:
            columns = dep["columns"]
            expression = dep["expression"]
            if not self._check_columns_in_real(columns, real_df):
                continue

            expression = self._strip_spaces(expression, columns)

            real_expression = self._sub_columns(expression, columns, "real_df")
            try:
                result = eval(real_expression)
                if not all(result):
                    continue
            except (SyntaxError, TypeError, NameError, ValueError):
                continue

            syn_expression = self._sub_columns(expression, columns, "syn_df")
            try:
                eval_result = eval(syn_expression)
            except (SyntaxError, TypeError, NameError, ValueError):
                continue
            syn_df = self._filter_rows_conservatively(
                syn_df,
                eval_result,
                "temporal_ordering",
                expression,
            )

        return syn_df.reset_index(drop=True)

    def _fix_uniqueness(
        self, uniqueness: dict, real_df: pd.DataFrame, syn_df: pd.DataFrame
    ):
        """Fix uniqueness dependencies across column combinations.

        Verifies that the combination of columns is unique in the real data.
        If so, duplicate synthetic combinations are filtered conservatively.
        This deliberately does not rewrite non-anchor columns from a real-data
        lookup: uniqueness of a column tuple is not a functional dependency
        from the anchor to every other column, and rewriting can collapse many
        synthetic rows into full duplicates.
        """
        for dep in uniqueness:
            columns = dep["columns"]

            if not self._check_columns_in_real(columns, real_df):
                continue

            if len(columns) < 2:
                continue

            if real_df.duplicated(subset=columns, keep=False).any():
                continue

            keep_mask = ~syn_df.duplicated(subset=columns, keep="first")
            syn_df = self._filter_rows_conservatively(
                syn_df,
                keep_mask,
                "uniqueness",
                dep.get("expression", f"unique({', '.join(columns)})"),
            )

        return syn_df.reset_index(drop=True)

    _PROTECTED_KEYWORDS = ["or", "and", "not", "in", "is"]

    def _strip_spaces(self, expression: str, columns: list[str]) -> str:
        """Remove cosmetic spaces from *expression* while preserving
        column names that contain spaces and Python keywords
        (``or``, ``and``, ``not``, ``in``, ``is``)."""
        sorted_cols = sorted(columns, key=len, reverse=True)
        placeholders = {}
        idx = 0

        for col in sorted_cols:
            if " " in col:
                ph = f"\x00PH{idx}\x00"
                placeholders[ph] = col
                expression = expression.replace(col, ph)
                idx += 1

        for kw in self._PROTECTED_KEYWORDS:
            ph = f"\x00PH{idx}\x00"
            placeholders[ph] = f" {kw} "
            expression = expression.replace(f" {kw} ", ph)
            idx += 1

        expression = expression.replace(" ", "")

        for ph, original in placeholders.items():
            expression = expression.replace(ph, original)

        return expression

    @staticmethod
    def _sub_columns(expression: str, columns: list[str], df_name: str) -> str:
        """Single-pass replacement of column names with ``df_name["col"]`` refs.

        Avoids substring collisions (e.g. ``lot_size`` inside
        ``lot_size_units``) by replacing all column names simultaneously
        via a regex alternation sorted longest-first.
        """
        sorted_cols = sorted(columns, key=len, reverse=True)
        pattern = "|".join(re.escape(c) for c in sorted_cols)
        return re.sub(pattern, lambda m: f'{df_name}["{m.group(0)}"]', expression)

    @staticmethod
    def _check_columns_in_real(columns: list[str], real_df: pd.DataFrame):
        """Return True if every column in *columns* exists in *real_df*."""
        for col in columns:
            if col not in real_df.columns:
                return False
        return True

    @staticmethod
    def _align_syn_series_dtype(
        real_series: pd.Series, syn_series: pd.Series
    ) -> pd.Series:
        """Coerce synthetic values to the real column dtype when possible."""
        target_dtype = real_series.dtype
        if pd.api.types.is_numeric_dtype(target_dtype):
            coerced = pd.to_numeric(syn_series, errors="coerce")
            if pd.api.types.is_integer_dtype(target_dtype):
                coerced = coerced.round()
            try:
                return coerced.astype(target_dtype)
            except (TypeError, ValueError):
                return coerced
        return syn_series

    @staticmethod
    def split_dependent_ranges_for_processing(
        deps: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """Split ``dependent_range`` items: short columns use rule-based fix, long use LLM refine."""
        short: list[dict] = []
        long: list[dict] = []
        for dep in deps:
            n = len(dep.get("columns") or [])
            if n >= 3:
                long.append(dep)
            else:
                short.append(dep)
        return short, long

    @staticmethod
    def anchor_samples_for_range(
        real_data: pd.DataFrame, dependent_range: dict
    ) -> pd.DataFrame:
        """Up to 5 random real rows per distinct anchor value (for encoding-checker context)."""
        anchor_column = dependent_range["anchor_column"]
        unique_anchors = real_data[anchor_column].unique()
        anchor_df = pd.DataFrame()
        for column in unique_anchors:
            single_anchor_df = real_data[real_data[anchor_column] == column]
            sample_size = min(5, len(single_anchor_df))
            if sample_size == 0:
                continue
            single_anchor_sample = single_anchor_df.sample(n=sample_size)
            anchor_df = pd.concat([anchor_df, single_anchor_sample])
        return anchor_df

    @classmethod
    def _encoding_checker_sample_chunks(
        cls, anchor_df: pd.DataFrame, anchor_column: str
    ) -> list[pd.DataFrame]:
        """Split full encoding-checker context by anchor values to fit model context."""
        if anchor_df.empty or anchor_column not in anchor_df.columns:
            return [anchor_df]

        unique_anchors = list(anchor_df[anchor_column].drop_duplicates())
        max_anchors = cls._ENCODING_CHECKER_MAX_ANCHORS_PER_RUN
        if len(unique_anchors) <= max_anchors:
            return [anchor_df]

        chunks: list[pd.DataFrame] = []
        for start in range(0, len(unique_anchors), max_anchors):
            anchors = unique_anchors[start : start + max_anchors]
            non_null_anchors = [anchor for anchor in anchors if pd.notna(anchor)]
            mask = anchor_df[anchor_column].isin(non_null_anchors)
            if any(pd.isna(anchor) for anchor in anchors):
                mask = mask | anchor_df[anchor_column].isna()
            chunks.append(anchor_df[mask])
        return chunks

    @staticmethod
    def _merge_encoding_checker_results(results: list[dict]) -> dict:
        """Combine EncodingChecker chunk outputs into one result."""
        if not results:
            return {
                "encoding_detected": False,
                "encoding_kind": "none",
                "confidence": 1.0,
                "readable_mapping": {},
            }

        readable_mapping: dict[str, dict] = {}
        kinds: set[str] = set()
        confidences: list[float] = []
        encoding_detected = False

        for result in results:
            if not isinstance(result, dict):
                continue
            encoding_detected = encoding_detected or bool(
                result.get("encoding_detected")
            )
            kind = result.get("encoding_kind")
            if kind and kind != "none":
                kinds.add(str(kind))
            try:
                confidences.append(float(result.get("confidence", 1.0)))
            except (TypeError, ValueError):
                pass

            chunk_mapping = result.get("readable_mapping") or {}
            if not isinstance(chunk_mapping, dict):
                continue
            for col, col_mapping in chunk_mapping.items():
                if not isinstance(col_mapping, dict):
                    continue
                readable_mapping.setdefault(col, {}).update(col_mapping)

        if not encoding_detected:
            encoding_kind = "none"
        elif len(kinds) == 1:
            encoding_kind = next(iter(kinds))
        else:
            encoding_kind = "mixed"

        return {
            "encoding_detected": encoding_detected,
            "encoding_kind": encoding_kind,
            "confidence": min(confidences) if confidences else 1.0,
            "readable_mapping": readable_mapping,
        }

    @staticmethod
    def _coerce_corrected_value(value: Any, target_dtype: Any) -> Any:
        """Cast detector corrections to the destination column dtype when possible."""
        try:
            if pd.api.types.is_float_dtype(target_dtype):
                return float(value)
            if pd.api.types.is_integer_dtype(target_dtype):
                return int(float(value))
            if pd.api.types.is_bool_dtype(target_dtype):
                if isinstance(value, str):
                    return value.strip().lower() in {"true", "1", "yes"}
                return bool(value)
            if pd.api.types.is_string_dtype(target_dtype) or target_dtype == object:
                return str(value)
        except (TypeError, ValueError):
            return value
        return value

    @staticmethod
    def _duplicate_labels_for_indices(
        df: pd.DataFrame, row_labels: list[Any] | set[Any]
    ) -> list[Any]:
        """Return labels from *row_labels* whose full rows duplicate any row in *df*."""
        if df.empty or not row_labels:
            return []
        duplicate_mask = df.duplicated(keep=False)
        return [
            label
            for label in row_labels
            if label in duplicate_mask.index and bool(duplicate_mask.loc[label])
        ]

    @staticmethod
    def _duplicate_retry_note(
        df: pd.DataFrame,
        duplicate_labels: list[Any],
        dependent_cols: list[str],
        max_context_rows: int = 20,
    ) -> str:
        """Build duplicate feedback for a retry on previously invalid rows only."""
        duplicate_context = df[df.duplicated(keep=False)].head(max_context_rows)
        forbidden_tuples = (
            duplicate_context[dependent_cols]
            .drop_duplicates()
            .to_dict(orient="records")
            if set(dependent_cols).issubset(duplicate_context.columns)
            else []
        )
        return (
            "\n\nDUPLICATE RETRY:\n"
            "The previous correction created full-row duplicates. Re-correct ONLY "
            "the rows in the new batch_rows; these rows were already marked invalid "
            "by your previous response.\n"
            "For every row in this retry, return is_valid=false with corrected_values "
            f"for these dependent columns in order: {dependent_cols}.\n"
            "Do NOT return corrected_values equal to any forbidden tuple below. "
            "Those tuples already produced duplicates.\n"
            f"forbidden corrected_values tuples: {json.dumps(forbidden_tuples, ensure_ascii=False)}\n"
            "Corrected rows must be dependency-valid and must not duplicate each "
            "other or any row in duplicate_context.\n"
            f"Duplicate row labels being retried: {list(duplicate_labels)}\n"
            "duplicate_context full rows:\n"
            f"{json.dumps(duplicate_context.to_dict(orient='records'), ensure_ascii=False)}\n"
        )

    @classmethod
    def _duplicate_retry_model_settings(
        cls, duplicate_retry_count: int
    ) -> ModelSettings | None:
        """Increase temperature only for duplicate-resolution retries."""
        if duplicate_retry_count <= 0:
            return None
        temperature = min(
            cls._DUPLICATE_RETRY_TEMPERATURE_MAX,
            cls._DUPLICATE_RETRY_TEMPERATURE_START
            + cls._DUPLICATE_RETRY_TEMPERATURE_STEP * (duplicate_retry_count - 1),
        )
        return {"temperature": temperature}

    @staticmethod
    def _duplicate_summary_text(df: pd.DataFrame) -> str:
        n_rows = len(df)
        n_unique = len(df.drop_duplicates()) if n_rows else 0
        return (
            f"rows={n_rows}, duplicate_rows={n_rows - n_unique}, "
            f"unique_rows={n_unique}"
        )

    async def llm_refine_dependent_ranges(
        self,
        fixed_syn_df: pd.DataFrame,
        dependent_ranges: list[dict],
        user_df_info: str,
        encoding_checker_model: Any,
        dependency_violation_detector_model: Any,
        *,
        real_df: pd.DataFrame | None = None,
        batch_size: int = 10,
        max_attempts: int = 3,
        langfuse_client: Any | None = None,
        langfuse_encoding_metadata: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """LLM pass: encoding check per range, then batched violation detection and in-place fixes."""
        real_df = self.real_df if real_df is None else real_df
        out_df = fixed_syn_df
        if self.verbose:
            seg = self._fixer_segment_label
            seg_part = f" [dim]· {seg} ·[/dim]" if seg else ""
            RICH_CONSOLE.print(
                f"\n[magenta]📋 LLM dependent_range refinement[/magenta]{seg_part} · "
                f"[dim]{len(dependent_ranges)} range(s), batch_size={batch_size}, "
                f"max_attempts={max_attempts}[/dim]"
            )

        summary: dict[str, Any] = {
            "dependent_range_count": len(dependent_ranges),
            "batch_size": batch_size,
            "max_attempts_per_batch": max_attempts,
            "max_duplicate_retries_per_batch": max(5, max_attempts),
            "encoding_check_runs": 0,
            "detector_batches_attempted": 0,
            "detector_batches_applied": 0,
            "detector_batches_skipped_invalid_response": 0,
            "detector_failed_attempts": 0,
            "detector_duplicate_retries": 0,
            "detector_duplicate_retry_temperatures": [],
            "detector_invalid_rows_dropped_after_duplicate_retries": 0,
            "model_cells_updated": 0,
        }

        status_cm = (
            RICH_CONSOLE.status(
                "[cyan]Dependent ranges…[/cyan]",
                spinner="dots",
            )
            if self.verbose
            else nullcontext()
        )
        with status_cm as range_status:
            n_ranges = len(dependent_ranges)
            for range_idx, dependent_range in enumerate(dependent_ranges, start=1):
                anchor_col_name = dependent_range.get("anchor_column")
                anchor_df = self.anchor_samples_for_range(real_df, dependent_range)
                anchor_column = dependent_range["anchor_column"]
                unique_anchor_count = (
                    anchor_df[anchor_column].nunique(dropna=False)
                    if anchor_column in anchor_df.columns
                    else 0
                )
                use_chunked_encoding = (
                    unique_anchor_count > self._ENCODING_CHECKER_MAX_ANCHORS_PER_RUN
                )

                if not use_chunked_encoding:
                    if self.verbose and range_status is not None:
                        range_status.update(
                            f"[cyan]Range {range_idx}/{n_ranges}[/cyan] · "
                            f"[yellow]{anchor_col_name}[/yellow] · "
                            f"{len(anchor_df)} context rows · encoding…"
                        )

                    enc_meta: dict[str, Any] = {
                        "agent": "EncodingCheckerAgent",
                        "range_index": range_idx,
                        "anchor_column": anchor_col_name,
                        "segment": self._fixer_segment_label,
                    }
                    if langfuse_encoding_metadata:
                        enc_meta.update(langfuse_encoding_metadata)
                    enc_span = langfuse_safe_trace(
                        langfuse_client,
                        name="EncodingCheckerAgent",
                        input_payload={
                            "range_index": range_idx,
                            "anchor_column": anchor_col_name,
                            "segment": self._fixer_segment_label,
                            "context_rows": len(anchor_df),
                        },
                        metadata=enc_meta,
                        new_trace=True,
                    )
                    try:
                        encoding_checker_agent = Agent(
                            name="EncodingCheckerAgent",
                            model=encoding_checker_model,
                            system_prompt=(
                                ENCODING_CHECKER_PROMPT.safe_substitute(
                                    dataset_info=user_df_info,
                                    sample=anchor_df.to_dict(orient="records"),
                                    dependency=dependent_range,
                                )
                            ),
                            instrument=False,
                        )

                        encoding_checker_run_result = await encoding_checker_agent.run()
                        self._accumulate_usage_summary(
                            encoding_checker_run_result, "EncodingCheckerAgent"
                        )
                        encoding_checker_result = (
                            encoding_checker_run_result.output.strip("```json\n").strip(
                                "\n```"
                            )
                        )
                        encoding_checker_result = json.loads(encoding_checker_result)
                        summary["encoding_check_runs"] += 1
                        _enc_out = langfuse_output_payload(
                            encoding_checker_result,
                            key="encoding_checker",
                        )
                        langfuse_safe_update(enc_span, output_payload=_enc_out)
                    except Exception as e:
                        langfuse_safe_update(
                            enc_span,
                            output_payload={"error": str(e)},
                            level="ERROR",
                            status_message=str(e),
                        )
                        raise
                    finally:
                        langfuse_safe_end(enc_span)
                else:
                    anchor_chunks = self._encoding_checker_sample_chunks(
                        anchor_df, anchor_column
                    )
                    if self.verbose and range_status is not None:
                        range_status.update(
                            f"[cyan]Range {range_idx}/{n_ranges}[/cyan] · "
                            f"[yellow]{anchor_col_name}[/yellow] · "
                            f"{len(anchor_df)} context rows · "
                            f"{len(anchor_chunks)} encoding call(s)…"
                        )

                    encoding_checker_results: list[dict] = []
                    for chunk_idx, anchor_chunk in enumerate(anchor_chunks, start=1):
                        if self.verbose and range_status is not None:
                            range_status.update(
                                f"[cyan]Range {range_idx}/{n_ranges}[/cyan] · "
                                f"[yellow]{anchor_col_name}[/yellow] · encoding "
                                f"[dim]{chunk_idx}/{len(anchor_chunks)}[/dim] · "
                                f"{len(anchor_chunk)} context rows"
                            )

                        enc_meta = {
                            "agent": "EncodingCheckerAgent",
                            "range_index": range_idx,
                            "chunk_index": chunk_idx,
                            "chunk_count": len(anchor_chunks),
                            "anchor_column": anchor_col_name,
                            "segment": self._fixer_segment_label,
                        }
                        if langfuse_encoding_metadata:
                            enc_meta.update(langfuse_encoding_metadata)
                        enc_span = langfuse_safe_trace(
                            langfuse_client,
                            name="EncodingCheckerAgent",
                            input_payload={
                                "range_index": range_idx,
                                "chunk_index": chunk_idx,
                                "chunk_count": len(anchor_chunks),
                                "anchor_column": anchor_col_name,
                                "segment": self._fixer_segment_label,
                                "context_rows": len(anchor_chunk),
                            },
                            metadata=enc_meta,
                            new_trace=True,
                        )
                        try:
                            encoding_checker_agent = Agent(
                                name="EncodingCheckerAgent",
                                model=encoding_checker_model,
                                system_prompt=(
                                    ENCODING_CHECKER_PROMPT.safe_substitute(
                                        dataset_info=user_df_info,
                                        sample=anchor_chunk.to_dict(orient="records"),
                                        dependency=dependent_range,
                                    )
                                ),
                                instrument=False,
                            )

                            encoding_checker_run_result = (
                                await encoding_checker_agent.run()
                            )
                            self._accumulate_usage_summary(
                                encoding_checker_run_result, "EncodingCheckerAgent"
                            )
                            encoding_checker_result = (
                                encoding_checker_run_result.output.strip(
                                    "```json\n"
                                ).strip("\n```")
                            )
                            encoding_checker_result = json.loads(
                                encoding_checker_result
                            )
                            encoding_checker_results.append(encoding_checker_result)
                            summary["encoding_check_runs"] += 1
                            _enc_out = langfuse_output_payload(
                                encoding_checker_result,
                                key="encoding_checker",
                            )
                            langfuse_safe_update(enc_span, output_payload=_enc_out)
                        except Exception as e:
                            langfuse_safe_update(
                                enc_span,
                                output_payload={"error": str(e)},
                                level="ERROR",
                                status_message=str(e),
                            )
                            raise
                        finally:
                            langfuse_safe_end(enc_span)

                    encoding_checker_result = self._merge_encoding_checker_results(
                        encoding_checker_results
                    )

                anchor_col = dependent_range["anchor_column"]
                dependent_cols = [
                    col for col in dependent_range["columns"] if col != anchor_col
                ]
                anchor_uniqs = out_df[anchor_col].unique()
                readable_mapping = encoding_checker_result.get("readable_mapping") or {}
                anchor_mapping = readable_mapping.get(anchor_col) or {}

                dep_cols = list(dependent_range["columns"])
                n_anchor_vals = len(anchor_uniqs)
                for anchor_idx, unq in enumerate(anchor_uniqs, start=1):
                    single_anchor_df = out_df[out_df[anchor_col] == unq]
                    encoded_unq = anchor_mapping.get(
                        str(unq), anchor_mapping.get(unq, unq)
                    )
                    total_batches = max(
                        1, (len(single_anchor_df) + batch_size - 1) // batch_size
                    )
                    if self.verbose and range_status is not None:
                        range_status.update(
                            f"[cyan]Range {range_idx}/{n_ranges}[/cyan] · "
                            f"[yellow]{anchor_col}[/yellow] · "
                            f"value [white]{unq!s}[/white] "
                            f"([dim]{anchor_idx}/{n_anchor_vals}[/dim]) · "
                            f"batch [dim]0/{total_batches}[/dim]"
                        )
                    per_anchor_batches = 0
                    per_anchor_skips = 0
                    per_anchor_cells = 0
                    for i in range(0, len(single_anchor_df), batch_size):
                        current_batch = (i // batch_size) + 1
                        if self.verbose and range_status is not None:
                            range_status.update(
                                f"[cyan]Range {range_idx}/{n_ranges}[/cyan] · "
                                f"[yellow]{anchor_col}[/yellow] · "
                                f"value [white]{unq!s}[/white] "
                                f"([dim]{anchor_idx}/{n_anchor_vals}[/dim]) · "
                                f"batch [dim]{current_batch}/{total_batches}[/dim]"
                            )
                        batch_full = single_anchor_df.iloc[i : i + batch_size].copy()
                        batch_dep = batch_full[dep_cols].copy()
                        per_anchor_batches += 1
                        current_batch_dep = batch_dep.copy()
                        pending_row_labels = list(current_batch_dep.index)
                        corrected_row_labels: set[Any] = set()
                        duplicate_retry_note = ""
                        n_fixes = 0
                        batch_applied = False
                        invalid_row_labels_for_batch: set[Any] = set()
                        detector_user_prompt = DEPENDENT_RANGE_BATCH_DETECTOR_USER_PROMPT
                        max_detector_attempts = max_attempts + max(5, max_attempts)
                        duplicate_retry_count = 0

                        for attempt in range(max_detector_attempts):
                            rows_to_check = current_batch_dep.loc[
                                pending_row_labels, dep_cols
                            ].copy()
                            if rows_to_check.empty:
                                batch_applied = True
                                break

                            summary["detector_batches_attempted"] += 1
                            attempt_prompt = DEPENDENT_RANGE_BATCH_DETECTOR_PROMPT.safe_substitute(
                                anchor_column=str(anchor_col),
                                anchor_value=str(unq),
                                anchor_encoded=str(encoded_unq),
                                dependent_columns=str(dependent_cols),
                                n_rows=str(len(rows_to_check)),
                                batch_rows=json.dumps(
                                    rows_to_check.to_dict(orient="records"),
                                    ensure_ascii=False,
                                ),
                            )
                            if duplicate_retry_note:
                                attempt_prompt += duplicate_retry_note
                            elif attempt > 0:
                                attempt_prompt += (
                                    DEPENDENT_RANGE_BATCH_DETECTOR_FORMAT_REMINDER
                                )

                            dependency_violation_detector_agent = Agent(
                                name="DependencyViolationDetectorAgent",
                                model=dependency_violation_detector_model,
                                system_prompt=attempt_prompt,
                                retries=2,
                                # High call volume: aggregate metrics only via
                                # ``DependencyFixer.batch_validation_summary`` in Langfuse.
                                instrument=False,
                            )

                            try:
                                run_model_settings = (
                                    self._duplicate_retry_model_settings(
                                        duplicate_retry_count
                                    )
                                )
                                if run_model_settings:
                                    summary[
                                        "detector_duplicate_retry_temperatures"
                                    ].append(float(run_model_settings["temperature"]))
                                run_result = (
                                    await dependency_violation_detector_agent.run(
                                        detector_user_prompt,
                                        model_settings=run_model_settings,
                                    )
                                )
                                self._accumulate_usage_summary(
                                    run_result, "DependencyViolationDetectorAgent"
                                )
                            except UnexpectedModelBehavior as e:
                                summary["detector_failed_attempts"] += 1
                                print(
                                    f"UnexpectedModelBehavior in DependencyViolationDetectorAgent "
                                    f"(attempt {attempt + 1}/{max_detector_attempts}, batch start row {i}): {e}"
                                )
                                if self.verbose:
                                    RICH_CONSOLE.print_exception(show_locals=False)
                                if attempt + 1 < max_detector_attempts:
                                    delay = min(
                                        60.0, (2**attempt) + random.uniform(0, 1.5)
                                    )
                                    await asyncio.sleep(delay)
                                continue
                            except json.JSONDecodeError as e:
                                summary["detector_failed_attempts"] += 1
                                print(
                                    f"JSONDecodeError in DependencyViolationDetectorAgent "
                                    f"(attempt {attempt + 1}/{max_detector_attempts}, batch start row {i}): {e}"
                                )
                                if self.verbose:
                                    RICH_CONSOLE.print_exception(show_locals=False)
                                if attempt + 1 < max_detector_attempts:
                                    delay = min(
                                        60.0, (2**attempt) + random.uniform(0, 1.5)
                                    )
                                    await asyncio.sleep(delay)
                                continue
                            except Exception as e:
                                summary["detector_failed_attempts"] += 1
                                print(
                                    f"Unexpected error in DependencyViolationDetectorAgent "
                                    f"(attempt {attempt + 1}/{max_detector_attempts}, batch start row {i}): "
                                    f"{type(e).__name__}: {e}"
                                )
                                if self.verbose:
                                    RICH_CONSOLE.print_exception(show_locals=False)
                                if attempt + 1 < max_detector_attempts:
                                    delay = min(
                                        60.0, (2**attempt) + random.uniform(0, 1.5)
                                    )
                                    await asyncio.sleep(delay)
                                continue
                            raw_output = (run_result.output or "").strip()
                            if raw_output.startswith("```"):
                                raw_output = "\n".join(raw_output.split("\n")[1:])
                                if raw_output.endswith("```"):
                                    raw_output = raw_output.rsplit("\n", 1)[0]
                                raw_output = raw_output.strip()

                            parsed_output = None
                            try:
                                parsed_output = json.loads(raw_output)
                            except json.JSONDecodeError:
                                json_start = raw_output.find("[")
                                json_end = raw_output.rfind("]")
                                if (
                                    json_start != -1
                                    and json_end != -1
                                    and json_end > json_start
                                ):
                                    try:
                                        parsed_output = json.loads(
                                            raw_output[json_start : json_end + 1]
                                        )
                                    except json.JSONDecodeError:
                                        parsed_output = None

                            is_valid_shape = isinstance(parsed_output, list) and all(
                                isinstance(item, dict) for item in parsed_output
                            )
                            if is_valid_shape and len(parsed_output) != len(rows_to_check):
                                is_valid_shape = False
                            if is_valid_shape:
                                dependency_violation_detector_result = parsed_output
                            else:
                                if attempt + 1 >= max_detector_attempts:
                                    break
                                continue

                            attempt_invalid_labels: list[Any] = []
                            attempt_fix_count = 0
                            for res_idx, res in enumerate(
                                dependency_violation_detector_result
                            ):
                                if res.get("is_valid"):
                                    continue

                                row_label = rows_to_check.index[res_idx]
                                attempt_invalid_labels.append(row_label)
                                invalid_row_labels_for_batch.add(row_label)
                                corrected_values = res.get("corrected_values") or []
                                if len(corrected_values) != len(dependent_cols):
                                    continue

                                for value_idx, value in enumerate(corrected_values):
                                    target_col = dependent_cols[value_idx]
                                    current_batch_dep.at[row_label, target_col] = (
                                        self._coerce_corrected_value(
                                            value, current_batch_dep[target_col].dtype
                                        )
                                    )
                                    attempt_fix_count += 1
                                corrected_row_labels.add(row_label)

                            if not corrected_row_labels and not attempt_invalid_labels:
                                batch_applied = True
                                break

                            candidate_out_df = out_df.copy()
                            candidate_out_df.loc[current_batch_dep.index, dep_cols] = (
                                current_batch_dep[dep_cols]
                            )
                            duplicate_labels = self._duplicate_labels_for_indices(
                                candidate_out_df,
                                corrected_row_labels.union(attempt_invalid_labels),
                            )
                            if not duplicate_labels:
                                out_df = candidate_out_df
                                n_fixes += attempt_fix_count
                                batch_applied = True
                                break

                            pending_row_labels = duplicate_labels
                            duplicate_retry_note = self._duplicate_retry_note(
                                candidate_out_df, duplicate_labels, dependent_cols
                            )
                            duplicate_retry_count += 1
                            summary["detector_duplicate_retries"] += 1

                        if not batch_applied:
                            labels_to_drop = list(
                                invalid_row_labels_for_batch or set(pending_row_labels)
                            )
                            labels_to_drop = [
                                label for label in labels_to_drop if label in out_df.index
                            ]
                            if labels_to_drop:
                                out_df = out_df.drop(index=labels_to_drop)
                            summary[
                                "detector_invalid_rows_dropped_after_duplicate_retries"
                            ] += len(labels_to_drop)

                        summary["detector_batches_applied"] += 1
                        summary["model_cells_updated"] += n_fixes
                        per_anchor_cells += n_fixes

                    if self.verbose:
                        skip_note = (
                            f" · [yellow]{per_anchor_skips} batch skip(s)[/yellow]"
                            if per_anchor_skips
                            else ""
                        )
                        RICH_CONSOLE.print(
                            f"  [green]✓[/green] [cyan]{anchor_col}[/cyan] = [white]{unq!s}[/white] · "
                            f"{len(single_anchor_df)} rows · "
                            f"[dim]{per_anchor_batches} batch(es), {per_anchor_cells} cell update(s)"
                            f"{skip_note}[/dim]"
                        )

        if self.verbose:
            print("\n[magenta]✓ LLM dependent_range refinement finished.[/magenta]")
        _emit_langfuse_batch_validation_summary(summary)
        return out_df

    def fix_dependencies(self):
        """Apply all dependency fixes to *syn_df* and return the corrected DataFrame.

        Dispatches each dependency type to the appropriate ``_fix_*`` method.
        For ``dependent_range``, rows with fewer than 3 columns in the dependency
        payload are fixed via :meth:`_fix_dependent_range`; wider dependencies are
        queued on ``self._pending_llm_dependent_ranges`` for
        :meth:`fix_dependencies_async`.
        """
        self._pending_llm_dependent_ranges = []
        syn_df = self.syn_df
        self._row_filter_baseline_count = len(syn_df)
        for dep_type, deps in self.dependencies.items():
            if dep_type == "mapping":
                syn_df = self._fix_mapping(deps, self.real_df, syn_df)
            elif dep_type == "rule":
                syn_df = self._fix_rule(deps, self.real_df, syn_df)
            elif dep_type == "range":
                syn_df = self._fix_range(deps, self.real_df, syn_df)
            elif dep_type == "correspondence":
                syn_df = self._fix_correspondence(deps, self.real_df, syn_df)
            elif dep_type == "logic":
                syn_df = self._fix_logic(deps, self.real_df, syn_df)
            elif dep_type == "temporal_ordering":
                syn_df = self._fix_temporal_ordering(deps, self.real_df, syn_df)
            elif dep_type == "uniqueness":
                syn_df = self._fix_uniqueness(deps, self.real_df, syn_df)
            elif dep_type == "dependent_range":
                short, long = self.split_dependent_ranges_for_processing(deps)
                self._pending_llm_dependent_ranges = long
                if short:
                    syn_df = self._fix_dependent_range(short, self.real_df, syn_df)

        return syn_df

    async def fix_dependencies_async(
        self,
        user_df_info: str,
        encoding_checker_model: Any,
        dependency_violation_detector_model: Any,
        *,
        real_df: pd.DataFrame | None = None,
        batch_size: int = 10,
        max_attempts: int = 3,
        verbose: bool = False,
        segment_label: str | None = None,
        langfuse_client: Any | None = None,
        langfuse_encoding_metadata: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Run :meth:`fix_dependencies` then LLM refinement for wide ``dependent_range`` items.

        *segment_label* is shown in logs to distinguish the main synthetic frame from the
        tail/outlier pool (e.g. ``\"Main synthetic (full body)\"`` vs ``\"Tail outliers\"``).
        """
        self.verbose = verbose
        self._fixer_segment_label = segment_label
        self._had_llm_dependent_range_pass = False
        self.llm_usage_summary = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }
        self.per_agent_usage_summary = {}
        role = f"[dim]· {segment_label} ·[/dim] " if segment_label else ""
        if self.verbose:
            RICH_CONSOLE.print(
                f"[bold cyan]DependencyFixer start[/bold cyan] {role}"
                f"input_rows={len(self.syn_df)}, "
                f"dependency_types={list(self.dependencies.keys())}, "
                f"duplicates=({self._duplicate_summary_text(self.syn_df)})"
            )
        syn_df = self.fix_dependencies()
        if self.verbose:
            RICH_CONSOLE.print(
                f"[cyan]Rule-based dependency fixes complete[/cyan] {role}"
                f"{len(syn_df)} rows remain, "
                f"duplicates=({self._duplicate_summary_text(syn_df)})."
            )
        pending = self._pending_llm_dependent_ranges
        if pending:
            syn_df = await self.llm_refine_dependent_ranges(
                fixed_syn_df=syn_df,
                dependent_ranges=pending,
                user_df_info=user_df_info,
                encoding_checker_model=encoding_checker_model,
                dependency_violation_detector_model=dependency_violation_detector_model,
                real_df=real_df,
                batch_size=batch_size,
                max_attempts=max_attempts,
                langfuse_client=langfuse_client,
                langfuse_encoding_metadata=langfuse_encoding_metadata,
            )
            self._had_llm_dependent_range_pass = True
        if self.verbose:
            RICH_CONSOLE.print(
                f"[bold green]DependencyFixer finished[/bold green] {role}"
                f"output_rows={len(syn_df)}, llm_pass={self._had_llm_dependent_range_pass}, "
                f"duplicates=({self._duplicate_summary_text(syn_df)})"
            )
        self._fixer_segment_label = None
        return syn_df
