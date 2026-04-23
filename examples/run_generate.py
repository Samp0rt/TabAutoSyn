"""Example script for running TabAutoSyn pipeline on a CSV dataset."""

import argparse
import asyncio
from pathlib import Path

import pandas as pd

from tabautosyn import TabAutoSyn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TabAutoSyn.generate() on a tabular CSV dataset."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to input CSV with real data.",
    )
    parser.add_argument(
        "--target-column",
        required=True,
        help="Name of target column used for class checks and curation.",
    )
    parser.add_argument(
        "--output-dir",
        default="tabautosyn_logs/example_run",
        help=(
            "Directory where final synthetic CSV and markdown summary are saved "
            "(passed to generate() as ouput_dir)."
        ),
    )
    parser.add_argument(
        "--output-csv-name",
        default="synthetic_data.csv",
        help="Filename for saved DataFrame returned by generate().",
    )
    parser.add_argument(
        "--separator",
        default=",",
        help="CSV separator for input file.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM temperature for agent prompts.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16000,
        help="Max tokens for LLM responses.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry count for dependency discovery agent.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout (seconds) for model calls.",
    )
    parser.add_argument(
        "--optimization-trials",
        type=int,
        default=None,
        help="Optional Optuna trial count for synthcity plugins.",
    )
    parser.add_argument(
        "--params",
        default=None,
        help="Optional path to precomputed optimization params (.pkl).",
    )
    parser.add_argument(
        "--user-df-info",
        default=None,
        help=(
            "Optional dataset description in one-line format, e.g. "
            "'us_location: US geographic dataset ...'. "
            "If omitted, generate() will infer it automatically."
        ),
    )
    parser.add_argument(
        "--save-pipeline-summary",
        action="store_true",
        help=(
            "If set, save pipeline markdown summary and final synthetic artifact "
            "inside --output-dir."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose logs.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_df = pd.read_csv(args.input_csv, sep=args.separator)
    synthesizer = TabAutoSyn(model="task_specific", task="ml", verbose=not args.quiet)
    user_df_info = args.user_df_info

    synthetic_df = await synthesizer.generate(
        train_data=real_df,
        user_df_info=user_df_info,
        target_column=args.target_column,
        optimization_trials=args.optimization_trials,
        params=args.params,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=args.retries,
        timeout=args.timeout,
        save_pipeline_summary=args.save_pipeline_summary,
        ouput_dir=str(output_dir),
    )


if __name__ == "__main__":
    asyncio.run(main())
