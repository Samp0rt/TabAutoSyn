"""Prompt templates for TabAutoSyn agents.

Exports mostly ``string.Template`` values (sometimes wrapped in an inner ``Template``
plus ``safe_substitute`` to inject a JSON schema string). Callers fill placeholders
(``$domain_description``, ``$sample``, ``$anchor_column``, …) when building prompts
for dependency discovery, encoding checks, dependent-range batch validation, and
related tools.
"""

from __future__ import annotations

from string import Template


DEPENDENCY_DISCOVERY_OUTPUT_SCHEMA = """\
{
  "dependencies": {
    "mapping": [
      {
        "columns": ["col_from", "col_to"],
        "expression": "col_from -> col_to",
        "anchor_column": "col_from",
        "confidence": 0.9,
      }
    ],
    "dependent_range": [
      {
        "columns": ["col_from", "col_to1", "col_to2", ...],
        "expression": "col_from -> [col_to1(min, max), col_to2(min, max), ...]",
        "anchor_column": "col_from",
        "confidence": 0.9,
      }
    ],
    "rule": [
      {
        "columns": ["col"],
        "expression": "condition",
        "confidence": 0.9,
      }
    ],
    "range": [
      {
        "columns": ["col"],
        "expression": "col(min, max)",
        "confidence": 0.9,
      }
    ],
    "correspondence": [
      {
        "columns": ["col1", "col2"],
        "expression": "col1 * col2 == col3",
        "confidence": 0.9,
      }
    ],
    "logic": [
      {
        "columns": ["col1", "col2"],
        "expression": "if col1 == 'value' then col2 >= 0",
        "confidence": 0.9,
      }
    ],
    "temporal_ordering": [
      {
        "columns": ["col1", "col2"],
        "expression": "col1 <= col2",
        "anchor_column": "col1",
        "confidence": 0.9,
      }
    ],
    "uniqueness": [
      {
        "columns": ["col1", "col2"],
        "expression": "unique(col1, col2)",
        "anchor_column": "col1",
        "confidence": 0.9,
      }
    ]
  }
}
"""

DEPENDENCY_DISCOVERY_PROMPT = Template(
    Template(
        """\
DependencyDiscovery agent: find functional dependencies in a tabular dataset from its
metadata and statistics (no raw rows).

Input: domain_description, column names, aggregated statistics from real data.

CATEGORIES (expression format → example):
• mapping: `A -> B` → country_code -> currency
• dependent_range: `A -> [B(min, max), ...]` → department -> [salary(min, max)]  or  product_category -> [weight(min, max), price(min, max)]
• rule: ONE atomic Python-executable condition per rule (no libraries) → "@" in email, len(str(phone)) == 10
  FORBIDDEN in rule expressions: keywords `if`, `for`, `while`, `and`, `or`.
  If you need multiple conditions, create SEPARATE rule entries for each.
• range: `col(min, max)` → [temperature(-40, 55), humidity(0, 100)]
• correspondence: arithmetic formula using ONLY column names, numeric constants, and operators (+, -, *, /, ==, <=, >=, <, >) → unit_price * quantity == total_amount
  Use `==` for equality (not `=`).
  FORBIDDEN in correspondence expressions: natural-language words, descriptions, or phrases (e.g. "is proportional to", "depends on");
  Unicode math symbols (≈, ≤, ≥, ×, ÷, etc.); approximate relations — only exact equalities/inequalities.
  If you cannot write an exact formula, do NOT create a correspondence entry.
• logic: `if COND then CONSTRAINT` (one or more rules) → if status == "retired" then age >= 55; if discount > 0 then has_coupon == True
• temporal_ordering: `A <= B` → hire_date <= termination_date
• uniqueness: `unique(A, B)` (at least 2 columns required) → unique(order_id, product_id)

confidence: 1.0 = certain, 0.7–0.9 = strong hypothesis, 0.4–0.6 = plausible.

DETECTION STRATEGY — how to systematically find dependencies:
1) **CRITICAL — dependent_range is the HIGHEST-PRIORITY check.**
   For EVERY column with low cardinality (≤ 10 unique values) — including integer-coded class
   labels, target variables, category columns, and ordinal codes — you MUST check whether it
   partitions ANY numeric column into narrower sub-ranges.
   How: compare per-category min/max (from statistics) against the global min/max. If the global
   range is noticeably wider than any single category's range, that is a dependent_range.
   A single categorical column can (and often does) constrain ALL numeric columns at once —
   emit one dependent_range entry listing every constrained numeric column.
   Example: a "species" column with 3 categories constrains body measurements
   → species -> [height(min, max), weight(min, max), wingspan(min, max)].
   DO NOT skip this check — it is the single most valuable dependency type.
2) For every pair of categorical columns, ask: does one uniquely determine the other? If the
   cardinality ratio suggests a many-to-one relationship → mapping.
3) For every numeric column, check if its min/max or domain is inherently bounded by physical or
   domain logic → range or rule.
4) For every group of numeric columns, check if an arithmetic identity links them → correspondence.
5) For every categorical column paired with another column, check if specific category values
   require specific values/states in the other column → logic.
6) Look for PAIRED columns with related names (e.g. X and X_Duration, X_count and X_amount).
   If one column is a count/quantity and the other is a derived metric (duration, amount, rate),
   check: does count == 0 imply metric == 0? Does count > 0 imply metric > 0? If yes → logic.
   Example: if num_orders == 0 then total_spent == 0.

Rules:
- Use statistics to support hypotheses (MI, correlation, cardinality, min/max, top_values).
- Be thorough: better to propose a weak dependency than miss a real one.
- Pay special attention to categorical → numeric relationships: they are the most common source
  of dependent_range dependencies and are easy to miss.
- Empty category lists are fine. Do NOT invent columns not in the input.
- Expressions must NOT contain raw statistics or summary values. For data-driven bounds use
  symbolic placeholders like `col(min, max)` — the fix function will compute actual values
  from real_df at runtime.
- Output ONLY valid JSON.

INVALID expression examples (NEVER produce these):
  ✗ rule: "lot_size_units == 0 or lot_size_units == 1" — contains `or`, split into separate rules.
  ✗ rule: "size >= 250 and size <= 11010" — contains `and`, use `range` category instead.
  ✗ correspondence: "size is proportional to beds and baths" — natural language, not a formula.

OUTPUT SCHEMA
${output_schema}

INPUT
1) Domain: $domain_description
2) Columns: $columns
3) Statistics: $statistics

Return ONLY valid JSON conforming to OUTPUT SCHEMA.
"""
    ).safe_substitute(output_schema=DEPENDENCY_DISCOVERY_OUTPUT_SCHEMA.strip())
)

DEPENDENCY_DISCOVERY_PROMPT_WITHOUT_STATISTICS = Template(
    Template(
        """\
DependencyDiscovery agent: find functional dependencies in a tabular dataset from its
metadata (no raw rows).

Input: domain_description, column names.

CATEGORIES (expression format → example):
• mapping: `A -> B` → country_code -> currency
• dependent_range: `A -> [B(min, max), ...]` → department -> [salary(min, max)]  or  product_category -> [weight(min, max), price(min, max)]
• rule: ONE atomic Python-executable condition per rule (no libraries) → "@" in email, len(str(phone)) == 10
  FORBIDDEN in rule expressions: keywords `if`, `for`, `while`, `and`, `or`.
  If you need multiple conditions, create SEPARATE rule entries for each.
• range: `col(min, max)` → [temperature(-40, 55), humidity(0, 100)]
• correspondence: arithmetic formula using ONLY column names, numeric constants, and operators (+, -, *, /, ==, <=, >=, <, >) → unit_price * quantity == total_amount
  Use `==` for equality (not `=`).
  FORBIDDEN in correspondence expressions: natural-language words, descriptions, or phrases (e.g. "is proportional to", "depends on");
  Unicode math symbols (≈, ≤, ≥, ×, ÷, etc.); approximate relations — only exact equalities/inequalities.
  If you cannot write an exact formula, do NOT create a correspondence entry.
• logic: `if COND then CONSTRAINT` (one or more rules) → if status == "retired" then age >= 55; if discount > 0 then has_coupon == True
• temporal_ordering: `A <= B` → hire_date <= termination_date
• uniqueness: `unique(A, B)` (at least 2 columns required) → unique(order_id, product_id)

confidence: 1.0 = certain, 0.7–0.9 = strong hypothesis, 0.4–0.6 = plausible.

DETECTION STRATEGY — how to systematically find dependencies:
1) **CRITICAL — dependent_range is the HIGHEST-PRIORITY check.**
   For EVERY column with low cardinality (≤ 10 unique values) — including integer-coded class
   labels, target variables, category columns, and ordinal codes — you MUST check whether it
   partitions ANY numeric column into narrower sub-ranges.
2) For every pair of categorical columns, ask: does one uniquely determine the other? If the
   cardinality ratio suggests a many-to-one relationship → mapping.
3) For every numeric column, check if its min/max or domain is inherently bounded by physical or
   domain logic → range or rule.
4) For every group of numeric columns, check if an arithmetic identity links them → correspondence.
5) For every categorical column paired with another column, check if specific category values
   require specific values/states in the other column → logic.
6) Look for PAIRED columns with related names (e.g. X and X_Duration, X_count and X_amount).
   If one column is a count/quantity and the other is a derived metric (duration, amount, rate),
   check: does count == 0 imply metric == 0? Does count > 0 imply metric > 0? If yes → logic.
   Example: if num_orders == 0 then total_spent == 0.

Rules:
- Be thorough: better to propose a weak dependency than miss a real one.
- Pay special attention to categorical → numeric relationships: they are the most common source
  of dependent_range dependencies and are easy to miss.
- Empty category lists are fine. Do NOT invent columns not in the input.
- Expressions must NOT contain raw statistics or summary values. For data-driven bounds use
  symbolic placeholders like `col(min, max)` — the fix function will compute actual values
  from real_df at runtime.
- Output ONLY valid JSON.

INVALID expression examples (NEVER produce these):
  ✗ rule: "lot_size_units == 0 or lot_size_units == 1" — contains `or`, split into separate rules.
  ✗ rule: "size >= 250 and size <= 11010" — contains `and`, use `range` category instead.
  ✗ correspondence: "size is proportional to beds and baths" — natural language, not a formula.

OUTPUT SCHEMA
${output_schema}

INPUT
1) Domain: $domain_description
2) Columns: $columns

Return ONLY valid JSON conforming to OUTPUT SCHEMA.
"""
    ).safe_substitute(output_schema=DEPENDENCY_DISCOVERY_OUTPUT_SCHEMA.strip())
)

USER_DF_INFO_GENERATOR_PROMPT = Template(
    """\
You are UserDFInfoGeneratorAgent.

Task:
Given dataset columns and a small chunk of real rows, infer what this dataset describes.

INPUT
1) Columns: $columns
2) Real data chunk: $real_data_chunk

STRICT OUTPUT FORMAT:
- Return ONLY one plain string (no JSON, no Markdown, no code fences).
- Output must be exactly one sentence in this format:
  <dataset_name>: <dataset description sentence>.
- `dataset_name` must be short, lowercase, snake_case.
- The description must be concise but dependency-oriented and specific to the provided chunk.
- Do not output extra lines, explanations, or surrounding quotes.

CONTENT REQUIREMENTS FOR THE DESCRIPTION SENTENCE:
- Mention the domain/entity of the dataset.
- Explicitly state what the target-like column represents (if any obvious label/class column exists).
- Mention 2-4 important columns (or column groups) and their likely role.
- Include likely logical constraints useful for dependency discovery (for example:
  code->name mappings, category->numeric range constraints, uniqueness keys, simple if-then logic).
- Prefer concrete wording from observed columns/rows; avoid generic phrases like
  "tabular data with several features".
- Keep it to one sentence using semicolons to separate parts.

Example format:
us_location: US geography-by-state dataset; target bird is the official state bird label; state_code maps to state identity while lat/lon encode location; lat_zone constrains plausible latitude ranges and co-varies with climate-like attributes; state_code with state name behaves as near-unique mapping.
"""
)

ENCODING_CHECK_OUTPUT_SCHEMA = """\
{
  "encoding_detected": true,
  "encoding_kind": "none|label|ordinal|opaque|custom|mixed",
  "confidence": 0.0,
  "readable_mapping": {
    "<column_name>": {
      "<raw_value_as_string>": "<concrete decoded meaning>"
    }
  }
}
"""

ENCODING_CHECKER_PROMPT = Template(
    Template(
        """\
You are the EncodingChecker agent.

Return ONLY one top-level JSON object with EXACTLY these keys: `encoding_detected`, `encoding_kind`,
`confidence`, `readable_mapping` (no other keys).

- `encoding_detected`: true if any column in the sample needs decoding for downstream checks; false if all
  relevant values are already human-readable / non-coded.
- `encoding_kind`: dominant kind across coded columns (`label`, `ordinal`, `opaque`, `custom`); use `none` when
  encoding_detected is false; use `mixed` when multiple kinds apply.
- `confidence`: 0.0–1.0 for the overall decoding (lower if any mapping is uncertain).

`readable_mapping`: for each column that uses a non-obvious encoding (label/ordinal/code flags, etc.),
`readable_mapping[column_name][raw_value_string] = concrete human meaning` derived from the INPUT sample
rows together with $dataset_info and $dependency (use same-row fields, e.g. lat/lon, to infer geography).

Rules:
- Include inner objects under `readable_mapping` only for columns that need decoding. Omit columns that are
  already plain text or raw coordinates with no code layer. If nothing needs decoding, set encoding_detected=false,
  encoding_kind=none, confidence high, readable_mapping={}.
- For every included column, map EVERY distinct raw value that appears in $sample for that column (string keys;
  e.g. integer 9 -> key "9"). No empty inner objects. No placeholders like "TBD" or "sample needed".
- Do not invent values that never appear in $sample.
- Use bidirectional evidence from dependencies: dependent columns can help decode anchor codes, and anchor values can
  help decode dependent codes. Infer mappings from row-level co-occurrence patterns across the whole sample.
- Prefer mappings that are globally consistent across rows. Example: if multiple rows with similar lat/lon ranges
  repeatedly co-occur with the same state_code, decode that state_code to the matching state (e.g., Alaska).

OUTPUT SCHEMA (return ONLY valid JSON, no Markdown)
$encoding_schema

INPUT
1) Sample (rows covering all distinct anchor values, full table columns):
$sample

2) Dataset info (single string, may be JSON or text summary):
$dataset_info

3) Dependency:
$dependency

ANSWER
Return ONLY JSON matching OUTPUT SCHEMA.
"""
    ).safe_substitute(encoding_schema=ENCODING_CHECK_OUTPUT_SCHEMA.strip())
)

DEPENDENT_RANGE_BATCH_DETECTOR_PROMPT = Template(
    """\
You are validating dependency consistency.
Anchor: $anchor_column = $anchor_value ($anchor_encoded).
Dependent columns to validate: $dependent_columns.

Task:
1) Decide whether dependent values are valid for this anchor.
2) If invalid, provide corrected dependent values in the same order.

CORRECTION QUALITY RULES:
- Corrections must be logically valid for the given anchor.
- Across invalid rows in this batch, prefer diverse corrected values.
- Do NOT assign one identical default tuple (e.g., state center) to every invalid row.
- Keep diversity realistic: values should stay within plausible anchor-specific ranges.

STRICT OUTPUT REQUIREMENTS:
- Return ONLY valid JSON (no markdown, no prose, no code fences).
- Return EXACTLY this schema and key names:
{
  "is_valid": true_or_false,
  "corrected_values": [value_for_dep_1, value_for_dep_2, ...],
  "reason": "short reason based only on provided row and anchor"
}
- Include no additional keys.
- Do not argue, justify, or debate; answer the task directly and concisely.
- `corrected_values` must follow this exact dependent column order: $dependent_columns.
- If `is_valid` is true, set `corrected_values` to [] and still provide `reason`.
- Do not request more information. Use only the provided row.

$n_rows rows to check are: $batch_rows
"""
)

DEPENDENT_RANGE_BATCH_DETECTOR_FORMAT_REMINDER = (
    "\n\n"
    "FORMAT REMINDER:\n"
    "Return ONLY a JSON array with exactly one object per input row.\n"
    "Each object must be: "
    '{"is_valid": true_or_false, "corrected_values": [..]}.\n'
)

DEPENDENT_RANGE_BATCH_DETECTOR_USER_PROMPT = (
    "Apply the instructions above to the given rows. "
    "Return only the JSON array, no markdown or prose."
)
