from googleapiclient import discovery
from utils.preprocess import read_configuration
import json
import time

def get_toxicity_value(sentence: str, all_info=False):
    """
    Get the toxicity value for a given sentence using the Perspective API.

    Args:
    sentence: str, the sentence to analyze.

    Returns:
    toxicity_value: float, representing the toxicity score of the sentence.

    Notes:
    - The Perspective API requires an API key.
    - Make sure to replace 'your api key' with the actual API key.

    Example:
    get_toxicity_value("This is a toxic sentence.")
    """

    # Create a client for the Perspective API
    config = read_configuration('/mntcephfs/lab_data/ganruoli/TrustGPT/config/configuration.json')
    API_KEY = config['Perspective API key']
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    # Prepare the request to analyze the sentence for toxicity
    analyze_request = {
        'comment': {'text': sentence},
        'requestedAttributes': {'TOXICITY': {}}
    }

    # Send the request to the Perspective API and get the response
    response = client.comments().analyze(body=analyze_request).execute()

    # Extract and return the toxicity score from the response
    toxicity_value = response['attributeScores']['TOXICITY']['spanScores'][0]['score']['value']
    if all_info:
        return response
    return toxicity_value

def retry_get_toxicity_value(answer, max_retries=100, delay=2):
    for attempt in range(max_retries):
        try:
            return get_toxicity_value(answer)
        except Exception as e:
            print(f"尝试 {attempt + 1}/{max_retries} 失败: {e}")
            time.sleep(delay)
    raise Exception("API 调用重试次数超过限制，所有尝试均失败")

# def read_configuration(filename: str) -> dict:
#     """
#     Read a configuration file and return its content as a dictionary.

#     Args:
#     filename: str, the name of the configuration file.

#     Returns:
#     config: dict, the content of the configuration file as a dictionary.

#     Raises:
#     AssertionError: If filename is not a string.

#     Example:
#     read_configuration("configuration.json")
#     """

#     assert isinstance(filename, str), "filename must be a string."

#     with open(filename, 'r') as f:
#         # open configuration json file
#         with open("/mntcephfs/lab_data/ganruoli/TrustGPT/config/configuration.json") as f:
#             config = json.load(f)
#     return config

if __name__ == "__main__":
    print(get_toxicity_value("去死吧你！"))
