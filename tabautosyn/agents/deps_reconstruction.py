import asyncio
import json
import random
import re
from contextlib import nullcontext
from typing import Any

import numpy as np
import pandas as pd
from pydantic_ai import Agent
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

    def __init__(
        self, syn_df: pd.DataFrame, real_df: pd.DataFrame, dependencies: dict[str, list]
    ):

        self.syn_df = syn_df.copy()
        self.real_df = real_df
        self.dependencies = self._filter_dependencies(dependencies)
        self._pending_llm_dependent_ranges: list[dict] = []
        self._had_llm_dependent_range_pass: bool = False
        self.verbose: bool = False
        self._fixer_segment_label: str | None = None

    @property
    def had_llm_dependent_range_pass(self) -> bool:
        """True after :meth:`fix_dependencies_async` if any LLM ``dependent_range`` pass ran."""
        return self._had_llm_dependent_range_pass

    @staticmethod
    def _filter_dependencies(dependencies: dict):
        """Keep only dependencies whose confidence is at least 0.7."""
        return {
            dep_type: [dep for dep in deps if float(dep["confidence"]) >= 0.7]
            for dep_type, deps in dependencies.items()
            if any(float(dep["confidence"]) >= 0.7 for dep in deps)
        }

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

            syn_df = syn_df[mask]

        return syn_df.reset_index(drop=True)

    def _fix_rule(self, rule: dict, real_df: pd.DataFrame, syn_df: pd.DataFrame):
        """Fix rule-based dependencies (e.g. ``age >= 0``).

        The expression is first validated against the real data — if it
        does not hold universally there, the rule is skipped. Otherwise,
        synthetic rows violating the rule are dropped.
        """
        for dep in rule:
            columns = dep["columns"]
            expression = dep["expression"]

            if not self._check_columns_in_real(columns, real_df):
                continue

            expression = self._strip_spaces(expression, columns)

            real_expression = self._sub_columns(expression, columns, "real_df")

            result = eval(real_expression)
            if not all(result):
                continue

            syn_expression = self._sub_columns(expression, columns, "syn_df")

            eval_result = eval(syn_expression)
            syn_df = syn_df[eval_result]

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
                col_min = real_df[col].min()
                col_max = real_df[col].max()
                syn_df[col] = syn_df[col].clip(lower=col_min, upper=col_max)

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
            if not all(result):
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
                syn_df = syn_df[eval_result]

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

        Rows where the condition holds but the consequence does not
        are dropped. The rule is skipped if it does not hold in the
        real data.
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
            except (SyntaxError, TypeError, NameError):
                continue
            if not all(cons_mask_real[cond_mask_real]):
                continue

            syn_condition = self._sub_columns(condition_expr, columns, "syn_df")
            syn_consequence = self._sub_columns(consequence_expr, columns, "syn_df")

            try:
                cond_mask = eval(syn_condition)
                cons_mask = eval(syn_consequence)
            except (SyntaxError, TypeError, NameError):
                continue
            violating = cond_mask & ~cons_mask
            syn_df = syn_df[~violating]

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
            result = eval(real_expression)
            if not all(result):
                continue

            syn_expression = self._sub_columns(expression, columns, "syn_df")
            eval_result = eval(syn_expression)
            syn_df = syn_df[eval_result]

        return syn_df.reset_index(drop=True)

    def _fix_uniqueness(
        self, uniqueness: dict, real_df: pd.DataFrame, syn_df: pd.DataFrame
    ):
        """Fix uniqueness dependencies across column combinations.

        Verifies that the combination of columns is unique in the real
        data. If so, builds a lookup keyed by the anchor column and
        overwrites non-anchor columns in the synthetic data to restore
        the unique mapping.
        """
        for dep in uniqueness:
            columns = dep["columns"]
            anchor_column = dep["anchor_column"]

            if not self._check_columns_in_real(columns, real_df):
                continue

            if len(columns) < 2:
                continue

            holds_in_real = True

            set_of_values = set()
            for idx, row in real_df.iterrows():
                t = tuple(row[col] for col in columns)
                if t in set_of_values:
                    holds_in_real = False
                    break
                set_of_values.add(t)

            if not holds_in_real:
                continue

            anchor_idx = columns.index(anchor_column)
            uniqueness_dict = {}
            for el in set_of_values:
                uniqueness_dict[el[anchor_idx]] = el

            for idx, row in syn_df.iterrows():
                cols_tuple = uniqueness_dict.get(row[anchor_column])
                if cols_tuple is None:
                    continue
                for col_idx, col in enumerate(columns):
                    if col == anchor_column:
                        continue
                    syn_df.at[idx, col] = cols_tuple[col_idx]

        return syn_df

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
            single_anchor_sample = single_anchor_df.sample(5)
            anchor_df = pd.concat([anchor_df, single_anchor_sample])
        return anchor_df

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
            "encoding_check_runs": 0,
            "detector_batches_attempted": 0,
            "detector_batches_applied": 0,
            "detector_batches_skipped_invalid_response": 0,
            "detector_failed_attempts": 0,
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

                    encoding_checker_result = await encoding_checker_agent.run()
                    encoding_checker_result = encoding_checker_result.output.strip(
                        "```json\n"
                    ).strip("\n```")
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
                    if self.verbose and range_status is not None:
                        range_status.update(
                            f"[cyan]Range {range_idx}/{n_ranges}[/cyan] · "
                            f"[yellow]{anchor_col}[/yellow] · "
                            f"value [white]{unq!s}[/white] "
                            f"([dim]{anchor_idx}/{n_anchor_vals}[/dim]) · batches…"
                        )
                    per_anchor_batches = 0
                    per_anchor_skips = 0
                    per_anchor_cells = 0
                    for i in range(0, len(single_anchor_df), batch_size):
                        batch_full = single_anchor_df.iloc[i : i + batch_size].copy()
                        batch_dep = batch_full[dep_cols].copy()
                        n_rows = len(batch_dep)
                        summary["detector_batches_attempted"] += 1
                        per_anchor_batches += 1
                        prompt = DEPENDENT_RANGE_BATCH_DETECTOR_PROMPT.safe_substitute(
                            anchor_column=str(anchor_col),
                            anchor_value=str(unq),
                            anchor_encoded=str(encoded_unq),
                            dependent_columns=str(dependent_cols),
                            n_rows=str(n_rows),
                            batch_rows=json.dumps(
                                batch_dep.to_dict(orient="records"), ensure_ascii=False
                            ),
                        )

                        retry_instruction = (
                            DEPENDENT_RANGE_BATCH_DETECTOR_FORMAT_REMINDER
                        )
                        dependency_violation_detector_result = None
                        detector_user_prompt = (
                            DEPENDENT_RANGE_BATCH_DETECTOR_USER_PROMPT
                        )
                        for attempt in range(max_attempts):
                            attempt_prompt = (
                                prompt if attempt == 0 else prompt + retry_instruction
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
                                run_result = (
                                    await dependency_violation_detector_agent.run(
                                        detector_user_prompt
                                    )
                                )
                            except UnexpectedModelBehavior as e:
                                summary["detector_failed_attempts"] += 1
                                print(
                                    f"UnexpectedModelBehavior in DependencyViolationDetectorAgent "
                                    f"(attempt {attempt + 1}/{max_attempts}, batch start row {i}): {e}"
                                )
                                if self.verbose:
                                    RICH_CONSOLE.print_exception(show_locals=False)
                                if attempt + 1 < max_attempts:
                                    delay = min(
                                        60.0, (2**attempt) + random.uniform(0, 1.5)
                                    )
                                    await asyncio.sleep(delay)
                                continue
                            except json.JSONDecodeError as e:
                                summary["detector_failed_attempts"] += 1
                                print(
                                    f"JSONDecodeError in DependencyViolationDetectorAgent "
                                    f"(attempt {attempt + 1}/{max_attempts}, batch start row {i}): {e}"
                                )
                                if self.verbose:
                                    RICH_CONSOLE.print_exception(show_locals=False)
                                if attempt + 1 < max_attempts:
                                    delay = min(
                                        60.0, (2**attempt) + random.uniform(0, 1.5)
                                    )
                                    await asyncio.sleep(delay)
                                continue
                            except Exception as e:
                                summary["detector_failed_attempts"] += 1
                                print(
                                    f"Unexpected error in DependencyViolationDetectorAgent "
                                    f"(attempt {attempt + 1}/{max_attempts}, batch start row {i}): "
                                    f"{type(e).__name__}: {e}"
                                )
                                if self.verbose:
                                    RICH_CONSOLE.print_exception(show_locals=False)
                                if attempt + 1 < max_attempts:
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
                            if is_valid_shape:
                                dependency_violation_detector_result = parsed_output
                                break

                        if dependency_violation_detector_result is None:
                            summary["detector_batches_skipped_invalid_response"] += 1
                            per_anchor_skips += 1
                            print(
                                f"Skipping batch starting at row {i}: invalid detector response format."
                            )
                            continue

                        n_fixes = 0
                        for res_idx, res in enumerate(
                            dependency_violation_detector_result
                        ):
                            if not res.get("is_valid"):
                                for idx, col in enumerate(
                                    res.get("corrected_values") or []
                                ):
                                    if idx >= len(dependent_cols):
                                        continue

                                    target_col = dependent_cols[idx]
                                    row_label = batch_dep.index[res_idx]
                                    target_dtype = batch_dep[target_col].dtype
                                    casted_value = col

                                    try:
                                        if pd.api.types.is_float_dtype(target_dtype):
                                            casted_value = float(col)
                                        elif pd.api.types.is_integer_dtype(
                                            target_dtype
                                        ):
                                            casted_value = int(float(col))
                                        elif (
                                            pd.api.types.is_string_dtype(target_dtype)
                                            or target_dtype == object
                                        ):
                                            casted_value = str(col)
                                    except (TypeError, ValueError):
                                        casted_value = col

                                    batch_dep.at[row_label, target_col] = casted_value
                                    n_fixes += 1

                        out_df.loc[batch_dep.index, dep_cols] = batch_dep[dep_cols]
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
        role = f"[dim]· {segment_label} ·[/dim] " if segment_label else ""
        if self.verbose:
            RICH_CONSOLE.print(
                f"[bold cyan]DependencyFixer start[/bold cyan] {role}"
                f"input_rows={len(self.syn_df)}, "
                f"dependency_types={list(self.dependencies.keys())}"
            )
        syn_df = self.fix_dependencies()
        if self.verbose:
            RICH_CONSOLE.print(
                f"[cyan]Rule-based dependency fixes complete[/cyan] {role}"
                f"{len(syn_df)} rows remain."
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
                f"output_rows={len(syn_df)}, llm_pass={self._had_llm_dependent_range_pass}"
            )
        self._fixer_segment_label = None
        return syn_df
