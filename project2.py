import sys
import json
from driver import InfoExtraction

# with open("config.json", "r") as config_file:
#     config = json.load(config_file)

# API_KEY = config["api_key"]
# CX_ID = config["cx_id"]

def main(model, google_api_key, google_engine_id, google_gemini_api_key, r, t, q, k):
    inforExtraction = InfoExtraction(model, google_api_key, google_engine_id, google_gemini_api_key, r, t, q, k)
    inforExtraction.start()


if __name__ == "__main__":
    if len(sys.argv) != 9:
        print("Usage: python3 project2.py [-spanbert|-gemini] <google api key> <google engine id> <google gemini api key> <r> <t> <q> <k>")
        sys.exit(1)

    # First argument should be either -spanbert or -gemini
    model = sys.argv[1]
    if model not in ["-spanbert", "-gemini"]:
        print("Error: First argument must be either '-spanbert' or '-gemini'.")
        sys.exit(1)

    try:
        google_api_key = sys.argv[2]
        google_engine_id = sys.argv[3]
        google_gemini_api_key = sys.argv[4]
        r = int(sys.argv[5])  # Number of iterations (integer)
        
        t = sys.argv[6]
        if model == "-spanbert":
            t = float(t)  # Ensure it's a float
            if not (0 <= t <= 1):
                raise ValueError("Error: 't' must be a real number between 0 and 1.")

        q = sys.argv[7]  # Query (string)
        k = int(sys.argv[8])  # Number of tuples to extract (integer)

    except ValueError:
        print("Error: 'r' and 'k' must be integers.")
        sys.exit(1)

    main(model, google_api_key, google_engine_id, google_gemini_api_key, r, t, q, k)
