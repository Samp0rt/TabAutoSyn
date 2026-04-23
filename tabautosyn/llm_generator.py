import ast
import pandas as pd
import json
import random
import logging
from tqdm import tqdm
from openai import OpenAI
import re


def has_curly_brace(text: str) -> bool:
    """
    Check if a text string contains curly braces.

    Args:
        text (str): The text string to check for curly braces.

    Returns:
        bool: True if the text contains '{' or '}', False otherwise.
    """
    return "{" in text or "}" in text


_INVALID_VALUE_RE = re.compile(r'"\s*:\s*(,|\}|\])')


def has_empty_value(json_str: str) -> bool:
    return bool(_INVALID_VALUE_RE.search(json_str))


def extract_json_objects(text: str) -> list:
    """
    Extract all valid JSON objects from a text string using stack-based parsing.

    This function finds all complete JSON objects (enclosed in curly braces) within
    the input text and attempts to parse them. It uses a stack to track brace
    nesting levels and only extracts complete, balanced JSON objects.

    Args:
        text (str): The text string to extract JSON objects from.

    Returns:
        list: A list of successfully parsed JSON objects. Objects that fail to
              parse are silently skipped.
    """
    json_objects = []
    stack = []
    start_index = -1

    for i, char in enumerate(text):
        if char == "{":
            if not stack:
                start_index = i
            stack.append(char)
        elif char == "}":
            if stack:
                stack.pop()
                if not stack:
                    json_str = text[start_index : i + 1]
                    if has_empty_value(json_str):
                        break
                    try:
                        obj = json.loads(json_str)
                        json_objects.append(obj)
                    except json.JSONDecodeError:
                        pass

    return json_objects


def extract_json(input_string: str) -> list | None:
    """
    Extract a JSON array from a string using bracket matching.

    This function finds the first complete JSON array (enclosed in square brackets)
    in the input string and attempts to parse it using ast.literal_eval.

    Args:
        input_string (str): The string to extract JSON array from.

    Returns:
        list or None: The parsed JSON array if successful, None if no valid
                     array is found or parsing fails.
    """
    start_pos = input_string.find("[")
    if start_pos == -1:
        return None

    nesting_level = 0
    for i in range(start_pos, len(input_string)):
        if input_string[i] == "[":
            nesting_level += 1
        elif input_string[i] == "]":
            nesting_level -= 1

        if nesting_level == 0:
            end_pos = i + 1
            json_string = input_string[start_pos:end_pos]
            json_string.replace("\n", "")

            try:
                data_ = ast.literal_eval(json_string)
                return data_
            except:
                continue
    return None


class LLMGenerator:
    """
    A synthetic data generator using Large Language Models (LLMs).

    This class generates synthetic tabular data that mimics the characteristics
    of real data by using an LLM to learn patterns from provided samples and
    generate new synthetic samples in JSON format.
    """

    def __init__(
        self,
        gen_client: OpenAI,
        gen_model_nm: str,
        real_data: pd.DataFrame,
        cols: list,
        gen_temperature: int = 0.5,
        verbose: bool = False,
        target_column: str = None,
    ) -> None:
        """
        Initialize the LLM synthetic data generator.

        Args:
            gen_client (OpenAI): The LLM client instance for generating synthetic data.
            gen_model_nm (str): Name of the LLM model to use for generation.
            real_data (pd.DataFrame): The real dataset to learn patterns from.
            cols (list): List of column names to include in generation.
            gen_temperature (float, optional): Temperature for LLM generation.
                                              Higher values increase randomness.
                                              Defaults to 0.5.
            verbose (bool, optional): Whether to print detailed progress information
                during data processing and synthesis. Defaults to False.
            target_column (str, optional): Name of the target column for stratification.
                                          If provided, ensures diverse target values in examples.
        """
        self.gen_client = gen_client
        self.gen_model_nm = gen_model_nm
        self.verbose = verbose
        self.real_data = real_data
        real_data.reset_index(inplace=True, drop=True)

        self.cols = cols
        self.gen_temperature = gen_temperature
        self.target_column = target_column

        # Setup logger
        self.logger = logging.getLogger(__name__)

        # Prepare stratified groups if target_column is provided
        if self.target_column is not None:
            if self.target_column not in self.real_data.columns:
                raise ValueError(
                    f"Target column '{self.target_column}' not found in real_data columns: {list(self.real_data.columns)}"
                )
            # Group data by target values
            self.target_groups = {}
            for target_value in self.real_data[self.target_column].unique():
                self.target_groups[target_value] = self.real_data[
                    self.real_data[self.target_column] == target_value
                ].index.tolist()
            self.unique_targets_count = len(self.target_groups)

    def instruction(self, sample: str, batch_size: int) -> tuple:
        """
        Generate system and user prompts for LLM-based synthetic data generation.

        Creates structured prompts that instruct the LLM to generate synthetic
        data samples based on the provided real data examples.

        Args:
            sample (str): String representation of real data samples in JSON format.
            batch_size (int): Number of samples to request from the LLM.

        Returns:
            tuple: A tuple containing (system_prompt, user_prompt) for the LLM.
        """
        prompt_sys = """
   The ultimate goal is to produce accurate and convincing synthetic
    data given the user provided samples. You MUST generate EXACTLY the requested number of samples."""

        prompt_user = f"""Here are examples from real data: {sample}\n"""
        prompt_user += f"\n\n Generate EXACTLY {batch_size} synthetic samples that mimic the provided samples. DO NOT COPY the samples. You must generate exactly {batch_size} samples - no more, no less. The response should be formatted STRICTLY as a list in JSON format, which is suitable for direct use in data processing scripts such as conversion to a DataFrame in Python. No additional text or numbers should precede the JSON data."
        return prompt_sys, prompt_user

    def row2dict(self, rows: pd.DataFrame) -> str:
        """
        Convert DataFrame rows to a string representation of dictionaries.

        Transforms DataFrame rows into a list of dictionaries, then converts to
        string format suitable for LLM prompts. The DataFrame index is reset
        during processing.

        Args:
            rows (pd.DataFrame): DataFrame containing the rows to convert.

        Returns:
            str: String representation of a list of dictionaries containing
                 the row data with proper formatting.
        """
        rows.reset_index(inplace=True, drop=True)
        res = []

        for i in range(len(rows)):
            example_data = {}
            row = rows.iloc[i, :]
            for column in self.cols:
                example_data[column] = row[column]

            res.append(example_data)

        return str(res)

    def _get_stratified_samples(
        self, batch_size: int, target_usage_counter: dict
    ) -> pd.DataFrame:
        """
        Select samples with stratification by target column.

        Ensures that examples include diverse target values, prioritizing
        less frequently used target values.

        Args:
            batch_size (int): Number of samples to select.
            target_usage_counter (dict): Counter tracking how many times each target value has been used.

        Returns:
            pd.DataFrame: Selected rows with diverse target values.
        """
        if self.target_column is None or target_usage_counter is None:
            raise ValueError("Stratified sampling requires target_column to be set")

        selected_indices = []
        num_examples_needed = min(batch_size, len(self.real_data))

        # Calculate how many samples per target group (roughly equal distribution)
        samples_per_target = max(1, num_examples_needed // self.unique_targets_count)
        remaining_samples = num_examples_needed - (
            samples_per_target * self.unique_targets_count
        )

        # Sort targets by usage (less used first) to ensure diversity
        sorted_targets = sorted(
            target_usage_counter.items(),
            key=lambda x: (
                x[1],
                random.random(),
            ),  # Sort by usage count, then randomize ties
        )

        # Select samples from each target group
        for target_value, usage_count in sorted_targets:
            if len(selected_indices) >= num_examples_needed:
                break

            available_indices = [
                idx
                for idx in self.target_groups[target_value]
                if idx not in selected_indices
            ]

            if not available_indices:
                continue

            # Determine how many samples to take from this target group
            if remaining_samples > 0:
                take_count = samples_per_target + 1
                remaining_samples -= 1
            else:
                take_count = samples_per_target

            take_count = min(
                take_count,
                len(available_indices),
                num_examples_needed - len(selected_indices),
            )

            # Randomly sample from this target group
            sampled = random.sample(available_indices, take_count)
            selected_indices.extend(sampled)
            target_usage_counter[target_value] += take_count

        # If we still need more samples, fill randomly from remaining data
        if len(selected_indices) < num_examples_needed:
            remaining_needed = num_examples_needed - len(selected_indices)
            available_all = [
                idx for idx in self.real_data.index if idx not in selected_indices
            ]
            if available_all:
                additional = random.sample(
                    available_all, min(remaining_needed, len(available_all))
                )
                selected_indices.extend(additional)

        # Return selected rows
        return self.real_data.loc[
            selected_indices[:num_examples_needed], self.cols
        ].copy()

    def run(self, n_samples: int, batch_size: int) -> None:
        """
        Generate synthetic data using LLM for a batch of real data samples.

        Processes real data in batches, sending sample groups to the LLM for
        synthetic data generation. Each batch uses consecutive rows from the real
        data as examples, cycling through the dataset if needed. If target_column
        is provided, uses stratified sampling to ensure diverse target values.

        Args:
            n_samples (int): Total number of synthetic samples to generate.
            batch_size (int): Number of samples to generate per batch.

        Returns:
            list: A list of lists containing JSON objects with generated synthetic data.

        Raises:
            ValueError: If target_column is provided and number of unique targets
                       exceeds n_samples.
        """
        # Validate target stratification requirements
        if self.target_column is not None:
            if self.unique_targets_count > n_samples:
                raise ValueError(
                    f"Number of unique target values ({self.unique_targets_count}) "
                    f"exceeds requested number of samples ({n_samples}). "
                    f"Cannot ensure all target values are represented. "
                    f"Please increase n_samples to at least {self.unique_targets_count}."
                )

        res = []
        samples_generated = 0
        # Track which target values have been used for stratification
        target_usage_counter = None
        if self.target_column is not None:
            target_usage_counter = {target: 0 for target in self.target_groups.keys()}

        for j in tqdm(range(0, n_samples, batch_size)):
            # Calculate how many samples to generate in this batch
            remaining_samples = n_samples - samples_generated
            current_batch_size = min(batch_size, remaining_samples)

            # Select examples with stratification if target_column is provided
            if self.target_column is not None:
                sampled_rows = self._get_stratified_samples(
                    current_batch_size, target_usage_counter
                )
            else:
                # Use random sampling to see diverse parts of dataset
                max_start_idx = max(
                    0, len(self.real_data) - min(batch_size, len(self.real_data))
                )
                real_data_idx = (
                    random.randint(0, max_start_idx) if max_start_idx > 0 else 0
                )
                sampled_rows = self.real_data.loc[
                    real_data_idx : real_data_idx
                    + min(batch_size, len(self.real_data))
                    - 1,
                    self.cols,
                ].copy()

            sample = self.row2dict(sampled_rows)
            sys_info, user_info = self.instruction(sample, current_batch_size)

            for i in range(5):  # 5 iterations to recieve valid json (else skip)
                try:
                    resp_temp = self.gen_client.chat.completions.create(
                        model=self.gen_model_nm,
                        messages=[
                            {"role": "system", "content": sys_info},
                            {"role": "user", "content": user_info},
                        ],
                        temperature=self.gen_temperature,
                        n=1,
                        max_tokens=5000,
                    )

                    content = resp_temp.choices[0].message.content
                    if content is not None:
                        json_objects = extract_json_objects(content)
                        if json_objects == []:
                            continue
                        elif json_objects:
                            # Ensure we don't exceed the target number of samples
                            if len(json_objects) > current_batch_size:
                                json_objects = json_objects[:current_batch_size]
                            res.append(json_objects)
                            samples_generated += len(json_objects)
                            break
                except:
                    if self.verbose:
                        self.logger.warning(
                            f"No valid JSON objects found in response for batch starting at {j}"
                        )

        return res

    def isValid(self, s):
        """
        Check if a string has balanced curly braces.

        Uses a stack-based approach to verify that all opening braces '{'
        have corresponding closing braces '}' in the correct order.

        Args:
            s (str): The string to check for balanced braces.

        Returns:
            bool: True if braces are balanced, False otherwise.
        """
        stack = []
        match = {"{": "}"}
        for i in s:
            if i in ["{"]:
                stack.append(i)
            if i in ["}"]:
                stack.pop()
        return stack == []

    def _generate_additional_samples(self, additional_needed: int, batch_size: int):
        """
        Generate additional samples until the exact number of valid samples is reached
        or the maximum number of attempts is exhausted.
        """
        additional_records = []
        max_attempts = 10

        if additional_needed <= 0:
            return additional_records

        # Use a random sample from real data as example
        sample_idx = random.randint(0, len(self.real_data) - 1)
        sampled_rows = self.real_data.loc[
            sample_idx : sample_idx + min(batch_size, len(self.real_data)) - 1,
            self.cols,
        ].copy()
        sample = self.row2dict(sampled_rows)

        attempts = 0
        while len(additional_records) < additional_needed and attempts < max_attempts:
            attempts += 1
            remaining = additional_needed - len(additional_records)

            sys_info, user_info = self.instruction(sample, remaining)

            try:
                resp_temp = self.gen_client.chat.completions.create(
                    model=self.gen_model_nm,
                    messages=[
                        {"role": "system", "content": sys_info},
                        {"role": "user", "content": user_info},
                    ],
                    temperature=self.gen_temperature,
                    n=1,
                    max_tokens=5000,
                )

                content = resp_temp.choices[0].message.content
                if content is None:
                    if self.verbose:
                        self.logger.warning(
                            "Additional generation response content is None"
                        )
                    continue

                json_objects = extract_json_objects(content)
                if not json_objects:
                    if self.verbose:
                        self.logger.warning("No JSON objects extracted from response")
                    continue

                valid = []
                for obj in json_objects:
                    if self._is_valid_sample(obj):
                        valid.append(obj)

                if not valid:
                    if self.verbose:
                        self.logger.warning(
                            "All generated samples were invalid (NaN/empty)"
                        )
                    continue

                take = min(remaining, len(valid))
                additional_records.extend(valid[:take])

            except Exception as e:
                if self.verbose:
                    self.logger.warning(
                        f"Error during additional generation attempt {attempts}: {e}"
                    )

        if len(additional_records) < additional_needed and self.verbose:
            self.logger.warning(
                f"Requested {additional_needed} additional samples, "
                f"but only generated {len(additional_records)} valid ones "
                f"after {max_attempts} attempts"
            )

        return additional_records

    def _is_valid_sample(self, sample: dict) -> bool:
        """
        Check if generated sample is valid (no NaNs / missing required columns).
        """
        for col in self.cols:
            if col not in sample:
                return False
            val = sample[col]
            if val is None:
                return False
            if isinstance(val, float):
                if pd.isna(val):
                    return False
            if isinstance(val, str) and (
                val.strip() == "" or val.strip().lower() == "nan"
            ):
                return False

        return True

    def generate(self, n_samples: int, batch_size: int) -> pd.DataFrame:
        """
        Main method to generate synthetic data with exact sample count control.

        This is the primary public method that orchestrates the entire synthetic
        data generation process. It ensures that exactly n_samples are generated
        by handling cases where too few or too many samples are produced.

        Args:
            n_samples (int): Total number of synthetic samples to generate.
            batch_size (int): Number of samples to generate per batch.

        Returns:
            pd.DataFrame: DataFrame containing exactly n_samples of synthetic data
                         with the same column structure as the original real data.
        """
        syn_res = self.run(n_samples=n_samples, batch_size=batch_size)

        records = []
        for sublist in syn_res:
            if sublist:
                # Filter out records containing NaN values
                valid_records = [
                    record
                    for record in sublist
                    if not any(
                        pd.isna(val)
                        for val in record.values()
                        if isinstance(record, dict)
                    )
                ]
                records.extend(valid_records)

        # Track how many samples were filtered out
        filtered_count = sum(len(sublist) for sublist in syn_res if sublist) - len(
            records
        )
        if filtered_count > 0 and self.verbose:
            self.logger.warning(
                f"Filtered out {filtered_count} samples containing NaN values"
            )

        # Ensure we have exactly n_samples
        if len(records) < n_samples:
            if self.verbose:
                self.logger.warning(
                    f"Generated only {len(records)} valid samples, requested {n_samples}. Generating additional samples..."
                )
            # Generate additional samples to reach n_samples
            additional_needed = n_samples - len(records)
            additional_records = self._generate_additional_samples(
                additional_needed, batch_size
            )
            records.extend(additional_records)
        elif len(records) > n_samples:
            if self.verbose:
                self.logger.warning(
                    f"Generated {len(records)} samples, requested {n_samples}. Truncating to {n_samples}"
                )
            records = records[:n_samples]

        df_syn = pd.DataFrame(records, columns=list(self.real_data.columns))

        # Final validation check for any remaining NaN values
        if df_syn.isnull().values.any():
            if self.verbose:
                self.logger.warning(
                    f"Final DataFrame contains {df_syn.isnull().sum().sum()} NaN values. Removing affected rows..."
                )
            df_syn = df_syn.dropna()

            # If we lost too many rows, regenerate
            if len(df_syn) < n_samples:
                shortage = n_samples - len(df_syn)
                additional_records = self._generate_additional_samples(
                    shortage, batch_size
                )
                df_additional = pd.DataFrame(
                    additional_records, columns=list(self.real_data.columns)
                )
                df_syn = pd.concat([df_syn, df_additional], ignore_index=True).iloc[
                    :n_samples
                ]

        return df_syn
