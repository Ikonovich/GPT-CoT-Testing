import json
import openai
import os
import time

# Load your OpenAI API key
openai.api_key = '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'

directory_path = './Results\Primary Test Results\gpt-3.5-turbo\zero_shot_cot\stepwise/'

files = os.listdir(directory_path)

for file in files:
    # Load the data
    with open(os.path.join(directory_path, file), 'r') as f:
        data = json.load(f)

    # Filter the data
    filtered_data = [item for item in data if item['Final Answer'] == item['GT']]

    # Prepare the queries
    queries = [item['Query'] for item in filtered_data]
    answers = [item['Final Answer'] for item in filtered_data]
    gts = [item['GT'] for item in filtered_data]

    # Prepare a list to store the responses
    responses = []

    s = """
    {2 * 4 = 8}\n
    {3 + 8 = 11}\n
    {11 + 5 = 16}\n
    {Answer = 16}\n
    """

    # Send the queries to the GPT-3 model and store the responses
    delay = 1  # start delay at 1 second
    iterator = 0
    for query, answer, gt in zip(queries, answers, gts):
        if(iterator > 100):
            break
        iterator +=1
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", 
                    messages=[
                        {"role": "system", "content":("You are a math problem-solving machine. You take in simple arithmetic problems, and output each step required to solve this arithmetic problem in order of operations. Show each mathematical step one by one, and place each step in squiggly brackets. Do not provide any additional explanations. For example, if the problem is 3 + 2 * 4 + 5, the output should be: " + s)}, 
                        {"role": "user", "content": str(query)}
                    ]
                )
                response_text = response.choices[0].message['content'].strip()
                response_dict = {
                    "query": query,
                    "response": response_text,
                    "answer": answer,
                    "gt": gt
                }
                responses.append(response_dict)
                delay = 1  # reset delay
                break
            except:
                print("Rate limit exceeded, waiting...")
                time.sleep(delay)
                delay *= 2  # double the delay

    # Print the responses
    for response in responses:
        print(response)

    # Create a list of dictionaries for each response
    response_dict_list = [{"response": r} for r in responses]

    # Write responses to a JSON file
    with open(f"./Datasets/Stepwise_Extracted/responses_{file}.json", "w") as f:
        json.dump(response_dict_list, f, indent=4, sort_keys=True)

    print(f"Responses saved to responses_{file}.json")
