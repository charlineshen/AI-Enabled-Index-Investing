from openai import OpenAI

client = OpenAI()
import csv
import os
import argparse
import pandas as pd
from openai import OpenAI
from cli import chunk, embed, load, chat_agent

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

GRADING_RUBRIC_WITH_EXPECTED_ANSWER = """
You are a grading assistant. Your task is to evaluate the given response against the expected answer and assign a score between -1 and 1 using the following rubric:

Grading Scheme:
- Correct: 1 point. The response correctly includes all the relevant information, or correctly identifies that the source text does not include information as required by the question.
- Partially correct (with irrelevant information): 0.8 point. The response correctly includes all the relevant information, but additionally contains irrelevant information.
- Partially correct (incomplete): 0.5 point. The response correctly includes some relevant information, but still misses some information.
- Wrong (information not found): 0 point. The response does not contain any relevant information, e.g., says the source text does not contain required information while it does, or addresses some irrelevant topic.
- Wrong (hallucination): -1 point. The response contains untruthful or misleading facts that are not supported by the source text.

Provide a brief reason for the score.
5. In your evaluation response, make sure to follow this pattern -- "Score: <X>. Reason: <Y>." with <X> as a number score !!! and <Y> as a brief reason for the score.
"""


GRADING_RUBRIC = """
You will be given a response written for a question about some source text. Your task is to grade the response on its factual accuracy according to the source text.

Grading Scheme:
- Correct: 1 point. The response correctly includes all the relevant information, or correctly identifies that the source text does not include information as required by the question.
- Partially correct (with irrelevant information): 0.8 point. The response correctly includes all the relevant information, but additionally contains irrelevant information.
- Partially correct (incomplete): 0.5 point. The response correctly includes some relevant information, but still misses some information.
- Wrong (information not found): 0 point. The response does not contain any relevant information, e.g., says the source text does not contain required information while it does, or addresses some irrelevant topic.
- Wrong (hallucination): -1 point. The response contains untruthful or misleading facts that are not supported by the source text.

Evaluation Steps:
1. Read the source text carefully.
2. Read the response and compare it to the source text. Check if all the information in the response is supported by the source text.
3. Read the question and compare the response to both the source text and the question. Check if the response includes all the relevant information to answer the question, and if it contains irrelevant information.
4. Assign a score according to the provided grading scheme above, and provide concise reasoning.
5. In your evaluation response, make sure to follow this pattern -- "Score: <X>. Reason: <Y>." with <X> as a number score !!! and <Y> as a brief reason for the score.
"""

title = "MSCI_Select_ESG_Screened_Indexes_Methodology_20230519"
zip_name = 'books'

def read_sample_qa(input_file):
    questions = []
    expected_answers = []

    # Open and read the CSV file
    with open(input_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row

        for row in csv_reader:
            questions.append(row[1])  # Question is in the second column
            expected_answers.append(row[2])  # Expected Answer is in the third column


    # Print the number of questions and answers to verify they match
    print(f"\nNumber of questions: {len(questions)}")
    print(f"Number of expected answers: {len(expected_answers)}")
    return questions, expected_answers

def read_sample_q(input_file):
    questions = []
    # Open and read the CSV file
    with open(input_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row

        for row in csv_reader:
            questions.append(row[1])  # Question is in the second column

    # Print the number of questions and answers to verify they match
    print(f"\nNumber of questions: {len(questions)}")
    return questions


def load_source_text(text_file = 'inputs/books/MSCI_Select_ESG_Screened_Indexes_Methodology_20230519.txt'):
    with open(text_file) as f:
        input_text = f.read()
    return input_text, text_file

def eval(prompt):
    response = client.chat.completions.create(model="gpt-3.5-turbo",  # You can replace with "gpt-4" if you have access
    messages=[
        {"role": "system", "content": "You are an evaluator."},
        {"role": "user", "content": prompt}
    ])

    # Parse the response
    evaluation = response.choices[0].message.content
    return evaluation  # Return the GPT-generated score and explanation

def evaluate_wo_expected_answer(input_file='evaluator-data/sample_q.csv', output_file='evaluator-data/sample_q_evaluation.csv'):
    questions = read_sample_q(input_file)
    results = []
    total_score = 0

    for i in range(len(questions)):
        query = questions[i]
        print("================= query: ", query)
        # Get the query, chunk, and actual response from the chat agent
        query, chunk, response = chat_agent(zip_name, title, query)
        source_text, text_filename = load_source_text()

        # Evaluate the response using GPT
        prompt = f"""
            {GRADING_RUBRIC}

            Question: {query}
            Source Text: {source_text}
            Response: {response}

            Provide the score and reason for the evaluation:
            """
        evaluation = eval(prompt)
        print(f"Evaluation: {evaluation}")

        score = float(evaluation.split()[1][:-1])
        total_score += score

        # Store results
        results.append({
            "Question": query,
            "Source text filename": text_filename,
            "Actual Response": response,
            "Evaluation": evaluation,
            "Retrieved Chunks": chunk
        })

    # Save results to an Excel file
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Evaluation results saved to '{output_file}'.")
    print("Average score: ", total_score/len(questions))

def evaluate_w_expected_answer(input_file='evaluator-data/sample_qa.csv', output_file='evaluator-data/sample_qa_evaluation.csv'):
    questions, expected_answers = read_sample_qa(input_file)
    results = []
    total_score = 0

    for i in range(len(questions)):
        query = questions[i]
        expected_answer = expected_answers[i]

        # Get the query, chunk, and actual response from the chat agent
        query, chunk, actual_response = chat_agent(zip_name, title, query)
        print(f"Query: {query}\nExpected Answer: {expected_answer}\nResponse: {actual_response}")

        # Evaluate the response using GPT
        prompt = f"""
            {GRADING_RUBRIC_WITH_EXPECTED_ANSWER}

            Question: {query}
            Expected Answer: {expected_answer}
            Actual Response: {actual_response}

            Provide the score and reason for the evaluation:
            """
        evaluation = eval(prompt)
        print(f"Evaluation: {evaluation}")

        score = float(evaluation.split()[1][:-1])
        total_score += score

        # Store results
        results.append({
            "Question": query,
            "Expected Answer": expected_answer,
            "Actual Response": actual_response,
            "Evaluation": evaluation,
            "Retrieved Chunks": chunk
        })

    # Save results to an Excel file
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Evaluation results saved to {output_file}.")
    print("Average score: ", total_score/len(questions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate QA responses using GPT')

    # Define optional arguments for input and output files
    # parser.add_argument('--input_file', type=str, default='evaluator-data/sample_qa.csv', help='Path to the input CSV file containing questions and expected answers')
    # parser.add_argument('--output_file', type=str, default='evaluator-data/sample_qa_evaluation.csv', help='Path to save the evaluation results')
    parser.add_argument('--with_expected_answer', action='store_true', help='Use this flag to evaluate with expected answers')
    
    args = parser.parse_args()

    if args.with_expected_answer:
        print("Evaluating responses with expected answers...")
        evaluate_w_expected_answer()
        # evaluate_w_expected_answer(input_file=args.input_file, output_file=args.output_file)
    else:
        print("Evaluating responses without expected answers...")
        evaluate_wo_expected_answer()
        # evaluate_wo_expected_answer(input_file=args.input_file, output_file=args.output_file)