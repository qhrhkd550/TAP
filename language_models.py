import os
import time
from typing import Dict, List
import requests
import json as json_module

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
        print(f"[DEBUG] Initializing Ollama with model: {model_name}, endpoint: {self.API_HOST_LINK}")

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

        # Debug: Print request details
        print(f"\n[DEBUG] Ollama API Request:")
        print(f"  Model: {self.model_name}")
        print(f"  Endpoint: {self.API_HOST_LINK}")
        print(f"  Messages: {len(conv)} messages")
        print(f"  First message preview: {str(conv[0])[:100]}..." if conv else "  (empty)")

        for attempt in range(self.API_MAX_RETRY):
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

                print(f"\n[DEBUG] Attempt {attempt + 1}/{self.API_MAX_RETRY}")
                print(f"[DEBUG] Sending request to {self.API_HOST_LINK}")

                response = requests.post(
                    self.API_HOST_LINK,
                    json=json_data,
                    timeout=self.API_TIMEOUT
                )

                print(f"[DEBUG] Response status code: {response.status_code}")

                response.raise_for_status()
                resp_json = response.json()

                print(f"[DEBUG] Response keys: {list(resp_json.keys())}")
                print(f"[DEBUG] Full response preview: {str(resp_json)[:200]}...")

                # Ollama returns the response in 'message' -> 'content'
                if 'message' in resp_json and 'content' in resp_json['message']:
                    output = resp_json['message']['content']
                    print(f"[DEBUG] ✓ Successfully extracted from message.content")
                    print(f"[DEBUG] Output preview: {output[:200]}...")
                elif 'response' in resp_json:
                    # For generate API endpoint
                    output = resp_json['response']
                    print(f"[DEBUG] ✓ Successfully extracted from response")
                    print(f"[DEBUG] Output preview: {output[:200]}...")
                else:
                    print(f"[DEBUG] ✗ Unexpected response format!")
                    print(f"[DEBUG] Full response: {json_module.dumps(resp_json, indent=2)}")
                    output = self.API_ERROR_OUTPUT

                break

            except requests.exceptions.ConnectionError as e:
                print(f'[ERROR] Connection failed: {e}')
                print(f'[ERROR] Is Ollama running? Try: ollama serve')
                time.sleep(self.API_RETRY_SLEEP)
            except requests.exceptions.Timeout as e:
                print(f'[ERROR] Request timeout: {e}')
                time.sleep(self.API_RETRY_SLEEP)
            except requests.exceptions.RequestException as e:
                print(f'[ERROR] Request exception: {type(e).__name__}: {e}')
                if hasattr(e, 'response') and e.response is not None:
                    print(f'[ERROR] Response body: {e.response.text[:500]}')
                time.sleep(self.API_RETRY_SLEEP)
            except Exception as e:
                print(f'[ERROR] Unexpected exception: {type(e).__name__}: {e}')
                import traceback
                traceback.print_exc()
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)

        if output == self.API_ERROR_OUTPUT:
            print(f"[ERROR] All {self.API_MAX_RETRY} retry attempts failed!")

        return output

    def batched_generate(self,
                        convs_list: List[List[Dict]],
                        max_n_tokens: int,
                        temperature: float,
                        top_p: float = 1.0,):
        print(f"\n[DEBUG] Batched generate called with {len(convs_list)} conversations")
        results = [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]
        print(f"[DEBUG] Batched generate completed. Success rate: {sum(1 for r in results if r != self.API_ERROR_OUTPUT)}/{len(results)}")
        return results
