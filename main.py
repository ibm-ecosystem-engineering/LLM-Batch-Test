from langchain_ibm import WatsonxLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import time

# load inputs
load_dotenv()

testing_data = pd.read_excel(os.getenv("input_data_file"))

# filter our dataset for empty prompt parameters
testing_data = testing_data.dropna()

testing_data['llm input'] = 'None'
testing_data['llm output'] = 'None'
testing_data['response times'] = 'None'

testing_data.head(10)

# instantiate wml connection
wml_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": os.getenv("api_key")
}

project_id = os.getenv("project_id")

llm_model_id = "meta-llama/llama-3-70b-instruct"

# step 1 parameters
generate_parameters_1 = {
    "decoding_method": "sample",
    "min_new_tokens": 1,
    "max_new_tokens": 400,
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 1,
    "repetition_penalty": 1.1,
    "stop_sequences": ['}']
}

llm_model_1 = WatsonxLLM(apikey=wml_credentials['apikey'],
                         url=wml_credentials['url'],
                         project_id=project_id,
                         model_id=llm_model_id,
                         params=generate_parameters_1)

# watsonx.ai call with dynamic prompt parameters
def llm_call_1(prompt_parameters, template):
    # rename your prompt_parameters
    input_variables = []
    for i in range(len(prompt_parameters)):
        input_variables.append(f'prompt_parameter_{i+1}')
    prompt = PromptTemplate(input_variables=input_variables, template=template)
    print(f'Input vars: {input_variables}\n\n')
    llm_chain = prompt | llm_model_1
    # create invoke parameter which is a dictionary of your prompt parameters
    prompt_data = {}
    for i, param in enumerate(prompt_parameters):
        prompt_data[f'prompt_parameter_{i+1}'] = param
    print(prompt_data)
    prompt_results = llm_chain.invoke(prompt_data)
    return prompt_results

# load prompt template from your prompt.txt file
prompt_template_file = open("prompt.txt")
prompt_template = prompt_template_file.read()
prompt_template_file.close()

for index, row in testing_data.iterrows():
    # we can dynamically create a list of prompt parameters based on the number of columns in your df
    prompt_parameters = []
    for i in range(len(testing_data.columns)):
        param_name = f'prompt_parameter_{i+1}'
        if param_name in testing_data.columns:
            prompt_parameters.append(row[param_name])
    print(f'-------------testing llm input {index + 1}-------------\n')
    # print current prompt
    llm_input = prompt_template
    for i, parameter in enumerate(prompt_parameters):
        llm_input = llm_input.replace(f'{{prompt_parameter_{str(i+1)}}}', str(parameter))
    testing_data.at[index, 'llm input'] = llm_input
    print(f'1) llm input:\n\n{llm_input}\n')
    start = time.time()
    prompt_results = llm_call_1(prompt_parameters, prompt_template)
    end = time.time()
    # llm output logic
    testing_data.at[index, 'llm output'] = prompt_results
    print(f'2) llm output:\n\n{prompt_results}\n')
    # response time logic
    api_response_time = round(end - start, 2)
    testing_data.at[index, 'response times'] = api_response_time
    print(f'3) api response time:\n\n{str(api_response_time)}\n')
    
# write the dataframe to an excel file
writer = pd.ExcelWriter(os.getenv("output_data_file"), engine='xlsxwriter')
testing_data.to_excel(writer, index=False, sheet_name='Sheet1')
workbook = writer.book
worksheet = writer.sheets['Sheet1']
cell_format = workbook.add_format({'text_wrap': True, 'valign': 'top', 'align': 'left'})
for i, column in enumerate(testing_data.columns):
    worksheet.set_column(i, i, 40, cell_format)
worksheet.set_column(3, 3, 70, cell_format)
writer.close()
