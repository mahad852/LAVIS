import re
import json
import pandas

def extract_options(input_string):
    pattern = r"\(([a-zA-Z])\)\s+([^()]*)"
    matches = re.findall(pattern, input_string)
    return {label: text.strip() for label, text in matches}

data = []


questions_df = pandas.read_csv("../datasets/MMVP/Questions.csv")
questions_df = questions_df.reset_index()

for _, row in questions_df.iterrows():
    index, question, options, answer =  row["lndex"], row["Question"], row["Options"], row["Correct Answer"]

    question = question.replace('"', '')
    options = options.replace('"', '')
    answer = answer.replace('"', '')

    question += f" Choose from the following options: {options}"

    options_dict = extract_options(options)
    # answer = [answer, f"{answer} {options_dict[answer[1]]}", options_dict[answer[1]]]

    data.append({
        "index" : int(index),
        "question_id" : int(index),
        "image" : f"{index}.jpg",
        "question" : question,
        "answer" : answer
    })

            
with open("../datasets/MMVP/Questions.json", "w") as f:
    json.dump(data, f)