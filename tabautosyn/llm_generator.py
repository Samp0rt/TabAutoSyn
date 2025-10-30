import ast
import pandas as pd
import json
import random
import logging
from tqdm import tqdm
from openai import OpenAI


def has_curly_brace(text: str) -> bool:
    """
    Check if a text string contains curly braces.
    
    Args:
        text (str): The text string to check for curly braces.
        
    Returns:
        bool: True if the text contains '{' or '}', False otherwise.
    """
    return '{' in text or '}' in text


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
        if char == '{':
            if not stack:
                start_index = i  
            stack.append(char)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack: 
                    json_str = text[start_index:i+1]
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
    start_pos = input_string.find('[')  
    if start_pos == -1:
        return None  

    nesting_level = 0
    for i in range(start_pos, len(input_string)):
        if input_string[i] == '[':
            nesting_level += 1
        elif input_string[i] == ']':
            nesting_level -= 1
        
        if nesting_level == 0:  
            end_pos = i + 1  
            json_string = input_string[start_pos:end_pos]
            json_string.replace('\n','')
            
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
    def __init__(self, 
    gen_client: OpenAI, 
    gen_model_nm: str, 
    real_data: pd.DataFrame, 
    cols: list, 
    gen_temperature: int = 0.5,
    verbose: bool = False) -> None:
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
        """
        self.gen_client = gen_client
        self.gen_model_nm = gen_model_nm
        self.verbose = verbose
        self.real_data = real_data
        real_data.reset_index(inplace=True, drop=True)

        self.cols = cols
        self.gen_temperature = gen_temperature
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
    

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
    

    def run(self, n_samples: int, batch_size: int) -> None:
        """
        Generate synthetic data using LLM for a batch of real data samples.
        
        Processes real data in batches, sending sample groups to the LLM for
        synthetic data generation. Each batch uses consecutive rows from the real
        data as examples, cycling through the dataset if needed.
        
        Args:
            n_samples (int): Total number of synthetic samples to generate.
            batch_size (int): Number of samples to generate per batch.
            
        Returns:
            list: A list of lists containing JSON objects with generated synthetic data.
        """
        res = []
        samples_generated = 0

        for j in tqdm(range(0, n_samples, batch_size)):
            # Calculate how many samples to generate in this batch
            remaining_samples = n_samples - samples_generated
            current_batch_size = min(batch_size, remaining_samples)
            
            # Use real data samples as examples (cycling through if needed)
            real_data_idx = j % len(self.real_data)
            sampled_rows = self.real_data.loc[real_data_idx:real_data_idx + min(batch_size, len(self.real_data)) - 1, self.cols].copy()
            sample = self.row2dict(sampled_rows)
            sys_info, user_info = self.instruction(sample, current_batch_size)

            resp_temp = self.gen_client.chat.completions.create(
                model=self.gen_model_nm,
                messages=[
                    {"role": "system", "content": sys_info},
                    {"role": "user", "content": user_info}
                ],
                temperature=self.gen_temperature,
                n=1,
                max_tokens=5000
            )

            content = resp_temp.choices[0].message.content
            if content is not None:
                json_objects = extract_json_objects(content)
                if json_objects:
                    # Ensure we don't exceed the target number of samples
                    if len(json_objects) > current_batch_size:
                        json_objects = json_objects[:current_batch_size]
                    res.append(json_objects)
                    samples_generated += len(json_objects)
                else:
                    if self.verbose:
                        self.logger.warning(f'No valid JSON objects found in response for batch starting at {j}')
            else:
                if self.verbose:
                    self.logger.error('Response is None')

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
        stack=[]
        match={'{':'}'}
        for i in s:
            if i in ['{']:
                stack.append(i)
            if i in ['}']:
                stack.pop()
        return stack==[]
    

    def _generate_additional_samples(self, additional_needed: int, batch_size: int):
        """
        Generate additional samples when the initial generation falls short.
        
        This private method is called when the initial generation doesn't produce
        enough samples. It uses a random sample from the real data as an example
        and generates the exact number of additional samples needed.
        
        Args:
            additional_needed (int): Number of additional samples needed.
            batch_size (int): Batch size for generation.
            
        Returns:
            list: List of additional generated samples.
        """
        additional_records = []
        
        # Use a random sample from real data as example
        sample_idx = random.randint(0, len(self.real_data) - 1)
        sampled_rows = self.real_data.loc[sample_idx:sample_idx + min(batch_size, len(self.real_data)) - 1, self.cols].copy()
        sample = self.row2dict(sampled_rows)
        sys_info, user_info = self.instruction(sample, additional_needed)

        resp_temp = self.gen_client.chat.completions.create(
            model=self.gen_model_nm,
            messages=[
                {"role": "system", "content": sys_info},
                {"role": "user", "content": user_info}
            ],
            temperature=self.gen_temperature,
            n=1,
            max_tokens=5000
        )

        content = resp_temp.choices[0].message.content
        if content is not None:
            json_objects = extract_json_objects(content)
            if json_objects:
                # Take only the number we need
                additional_records = json_objects[:additional_needed]
            else:
                if self.verbose:
                    self.logger.warning(f'No valid JSON objects found in additional generation response')
        else:
            if self.verbose:
                self.logger.error('Additional generation response is None')
            
        return additional_records
    
    
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
                records.extend(sublist)

        # Ensure we have exactly n_samples
        if len(records) < n_samples:
            if self.verbose:
                self.logger.warning(f"Generated only {len(records)} samples, requested {n_samples}. Generating additional samples...")
            # Generate additional samples to reach n_samples
            additional_needed = n_samples - len(records)
            additional_records = self._generate_additional_samples(additional_needed, batch_size)
            records.extend(additional_records)
        elif len(records) > n_samples:
            if self.verbose:
                self.logger.warning(f"Generated {len(records)} samples, requested {n_samples}. Truncating to {n_samples}")
            records = records[:n_samples]

        df_syn = pd.DataFrame(records, columns=list(self.real_data.columns))
        
        return df_syn
