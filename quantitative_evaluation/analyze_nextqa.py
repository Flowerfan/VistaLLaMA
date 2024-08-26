import os
import argparse
import json
import ast
from multiprocessing.pool import Pool


def parse_args():
    parser = argparse.ArgumentParser(
        description="question-answer-generation-using-gpt-3"
    )
    parser.add_argument(
        "--pred_path", required=True, help="The path to file containing prediction."
    )
    # parser.add_argument("--api_key", required=True, help="OpenAI API key.")
    args = parser.parse_args()
    return args


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    file = open(args.pred_path)
    print(f"load prediction file from {args.pred_path}")
    pred_contents = json.load(file)

    # Calculate average score and accuracy
    score_sum = 0
    count = 0
    T_count = [0, 0, 0]
    C_count = [0, 0, 0]
    D_count = [0, 0, 0]
    yes_count = 0
    no_count = 0
    for key, result in pred_contents.items():
        # Computing score
        count += 1
        try:
            score_match = result[0]["score"]
        except:
            print(result)
            continue
        score = int(score_match)
        score_sum += score

        # Computing accuracy
        pred = result[0]["pred"]
        if "yes" in pred.lower():
            yes_count += 1
        elif "no" in pred.lower():
            no_count += 1
        if key.startswith('T'):
            T_count[0] += 1
            if "yes" in pred.lower():
                T_count[1] += 1
            elif "no" in pred.lower():
                T_count[2] += 1
        elif key.startswith('D'):
            D_count[0] += 1
            if "yes" in pred.lower():
                D_count[1] += 1
            elif "no" in pred.lower():
                D_count[2] += 1
        elif key.startswith('C'):
            C_count[0] += 1
            if "yes" in pred.lower():
                C_count[1] += 1
            elif "no" in pred.lower():
                C_count[2] += 1
            
        

    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print("Accuracy:", accuracy)
    print("T Accuracy:", T_count[1] / (T_count[1] + T_count[2]))
    print("D Accuracy:", D_count[1] / (D_count[1] + D_count[2]))
    print("C Accuracy:", C_count[1] / (C_count[1] + C_count[2]))
    print("Average score:", average_score)


if __name__ == "__main__":
    main()
