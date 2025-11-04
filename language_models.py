import os
import time
from typing import Dict, List
import requests

from config import OLLAMA_API_LINK


class LanguageModel():
    def __init__(self, model_name):
        self.model_name = model_name

    def batched_generate(self, prompts_list: List, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError


class Ollama(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 20
    API_TIMEOUT = 100

    def __init__(self, model_name, api_link=None):
        super().__init__(model_name)
        self.API_HOST_LINK = api_link or OLLAMA_API_LINK

    def generate(self, conv: List[Dict],
                max_n_tokens: int,
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT

        for _ in range(self.API_MAX_RETRY):
            try:
                # Ollama API uses chat format
                json_data = {
                    "model": self.model_name,
                    "messages": conv,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_predict": max_n_tokens,
                    }
                }

                response = requests.post(
                    self.API_HOST_LINK,
                    json=json_data,
                    timeout=self.API_TIMEOUT
                )

                response.raise_for_status()
                resp_json = response.json()

                # Ollama returns the response in 'message' -> 'content'
                if 'message' in resp_json and 'content' in resp_json['message']:
                    output = resp_json['message']['content']
                elif 'response' in resp_json:
                    # For generate API endpoint
                    output = resp_json['response']
                else:
                    print(f"Unexpected response format: {resp_json}")
                    output = self.API_ERROR_OUTPUT

                break

            except requests.exceptions.RequestException as e:
                print(f'Request exception: {type(e).__name__}: {e}')
                time.sleep(self.API_RETRY_SLEEP)
            except Exception as e:
                print(f'Exception: {type(e).__name__}: {e}')
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)

        return output

    def batched_generate(self,
                        convs_list: List[List[Dict]],
                        max_n_tokens: int,
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]
