import chromadb
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

OpenAI.api_key = os.getenv('OPENAI_API_KEY_ANALYTICS')
client = OpenAI()

data = {'ID': [1, 2, 3],
        'Name': ['A', 'B', 'C'],
        'Pass/Fail': [True, False, True]}

df = pd.DataFrame(data)

df = pd.read_csv("Traffic_Incidents.csv")
column_names = df.columns.to_list()
print(column_names)
first_row = df.iloc[0].to_list()
print(first_row)

prompt1 = '''
You are expert at making meaningful sentences out of column names of tabular data. I will give you two lists.
List one has column names of a dataframe. List two contains the first row 
of the tabular data stored in the same sequence as the column names from list one.
You have two tasks. 
First task is to form a meaningful sentence using context from list one and values from list two.
Second task is to take the sentence generated from task one and then
change all the list two values from the sentence with "df['corresponding value from list one']".
ONLY return the sentence and nothing else.
List one: {}
List two: {} 
'''

prompt2 = '''
You are expert at making meaningful sentences out of tabular data. I will give you two lists.
List one has column names of a dataframe. List two contains the first row 
of the tabular data stored in the same sequence as the column names from list one.
Your task is to form a meaningful sentence using using all the values from list one and two by
using list one for context and list two for values. The output should be in the form of a python f string
List one: {}
List two: {} 
'''

prompt = prompt1.format(column_names, first_row)

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",  # or other available engines
    messages=[
        {"role": "system", "content": prompt}
        ]
        )
print(completion.choices[0].message.content)


