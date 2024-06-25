<!-- ABOUT THE PROJECT -->

# LLM-Batch-Test

The LLM-Batch-Test repo contains a script that can be used to batch test llm inputs against a watsonx.ai instance. The script ingests an excel file with parameters that are inserted into a prompt template, uses the [IBM Watson Machine Learning Python SDK](https://ibm.github.io/watsonx-ai-python-sdk/foundation_models.html) to run llm calls, and outputs an excel file with the results. The goal of the tool is to reduce execution time for running tests.

<!-- GETTING STARTED -->

## Getting Started

### Prerequisites

The following prerequisites are required to run the script:

1. Python3
2. IBM Cloud api key (this must be for the same cloud account that hosts the watsonx.ai instance)
3. watsonx.ai project id

### Installation

1. Clone the repo

   ```bash
   git clone git@github.ibm.com:hcbt/LLM-Batch-Test.git
   ```

2. Change directory into LLM-Batch-Test

   ```bash
   cd LLM-Batch-Test
   ```

3. Create a python virtual environment

   ```bash
   python3 -m venv virtual-env
   source virtual-env/bin/activate
   pip3 install -r requirements.txt
   ```

4. Copy env file to .env

   ```bash
   cp env .env
   ```

5. Configure parameters in .env and set the values of your environment
   1. For your "input_data_file", see [Configuring Your Input Excel File](#configuring-your-input-excel-file)
   2. You can name your "output_data_file" anything
   3. Ensure both the input and output file names are of type .xlsx

6. Add your freefrom prompt to "prompt.txt"
   1. Ensure your prompt parameters are called "prompt_parameter_i" where i starts from 1
   2. prompt_parameter_1 corresponds to the values in column 1 of your "input_data_file", prompt_parameter_2 corresponds to the second column, and so on
   3. You can view the "sample_prompt.txt" file to get an idea of the required structure
   4. To include curly brackets in your prompt, escape them using double curlies, ex: {{

7. Run the following to start the script

   ```bash
   python3 main.py
   ```

8. Run the following command to exit the python virtual environment:

   ```bash
   deactivate
   ```

## Configuring your Input Excel File

The repository contains a "sample_llm_inputs" file that you can copy, edit, and use to test.

The input excel file must have a column per each prompt parameter you want to insert into your prompt template.

You can view the "sample_llm_inputs" and "sample_prompt" to understand the required configuration.

## Understanding the Results

You can observe real time results in the terminal. Each time a questions is asked, you can view the llm input, llm output, and response time.

When all tests are completed, an output excel file with your results is created using the name specified in your env file. The output file contains the inputted prompt parameters, the llm input, the llm output and response times.
