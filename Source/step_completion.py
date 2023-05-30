import json
import openai
import os
import time

# Load your OpenAI API key
openai.api_key = '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'



files = ["./Datasets/Stepwise_Extracted/Consequential/consequential_fixed.json", "./Datasets/Stepwise_Extracted/Inconsequential/inconsequential_fixed.json"]

for file in files:
    # Prepare a list to store the responses
    responses = []

    # Load the JSON file
    with open(file, 'r') as f:
        data = json.load(f)

    # Prepare an empty list to store the extracted data
    extracted_data = []

    # Loop over each item in the data
    for item in data:
        # Store the gt, query, and response
        gt = item['gt']
        query = item['query']
        response = item['response']

        # Concatenate the steps into a single string
        steps_string = '\n'.join(item['steps'])

        # Append the extracted data to the list
        extracted_data.append({
            'gt': gt,
            'query': query,
            'response': response,
            'steps_string': steps_string
        })
    response_text = ""
    # Send the queries to the GPT-3 model and store the responses
    delay = 1  # start delay at 1 second
    print("beginning.")
    iterator = 0
    for item in extracted_data:
        iterator+=1
        print(iterator)
        if(iterator > 4):
            break
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4", 
                    messages=[
                        {"role": "assistant", "content":str(item["steps_string"])},
                        {"role": "user", "content": str(item["query"])}
                    ]
                )
                response_text = response.choices[0].message['content'].strip()
                response_dict = {
                    "query": item["query"],
                    "steps": item["steps_string"],
                    "answer": item["response_text"],
                    "gt": item["gt"]
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
    print(response_dict_list)
    break
    # Write responses to a JSON file
    #with open(f"./Datasets/Stepwise_Extracted/responses_{file}.json", "w") as f:
        #json.dump(response_dict_list, f, indent=4, sort_keys=True)

    #print(f"Responses saved to responses_{file}.json")
