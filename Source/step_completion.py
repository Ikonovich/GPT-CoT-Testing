import json
import openai
import os
import time
import traceback


# Load your OpenAI API key
openai.api_key = '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'



files = ["consequential_fixed.json", "inconsequential_fixed.json"]
models = ["gpt-3.5-turbo", "gpt-4"]
for selected_model in models:
    for file in files:
        # Prepare a list to store the responses
        responses = []

        # Load the JSON file
        
        if(file == "consequential_fixed.json"):
            filepath =  "./Datasets/Stepwise_Extracted/Consequential/" + file
        else:
            filepath = "./Datasets/Stepwise_Extracted/Inconsequential/" + file
        with open(filepath, 'r') as f:
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
            steps_length = item["steps_length"]
            # Append the extracted data to the list
            extracted_data.append({
                'gt': gt,
                'query': query,
                'response': response,
                'steps_string': steps_string,
                'steps_length': steps_length
            })
        response_text = ""
        # Send the queries to the selected model and store the responses
        delay = 1  # start delay at 1 second
        for item in extracted_data:
            # While loop used to handle timeouts and rate errors
            while True:
                try:
                    response = openai.ChatCompletion.create(
                        model=selected_model, 
                        messages=[
                            {"role": "assistant", "content":str(item["steps_string"])},
                            {"role": "user", "content": str(item["query"])}
                        ]
                    )
                    response_text = response.choices[0].message['content'].strip()
                    response_dict = {
                        "query": item["query"],
                        "steps": item["steps_string"],
                        "answer": response_text,
                        "gt": item["gt"],
                        "steps_length": item["steps_length"]
                    }
                    responses.append(response_dict)
                    delay = 1  # reset delay
                    break
                except Exception as e:
                    print("Exception Occured.\nWaiting", delay, "seconds...\n")
                    time.sleep(delay)
                    delay *= 2  # double the delay
            

        # Create a list of dictionaries for each response
        response_dict_list = [{"response": r} for r in responses]

        # Write responses to a JSON file
        with open(f"./Results/Secondary Test Results/{selected_model}/stepwise-step-randomization/responses_{file}.json", "w") as f:
            json.dump(response_dict_list, f, indent=4, sort_keys=True)

        print(f"Responses saved to responses_{file}.json")
