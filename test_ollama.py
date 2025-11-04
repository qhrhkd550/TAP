#!/usr/bin/env python3
"""
Test script to verify Ollama connectivity and response format
"""

import requests
import json

def test_ollama_connection():
    """Test basic Ollama connectivity"""
    print("="*60)
    print("OLLAMA CONNECTIVITY TEST")
    print("="*60)

    endpoint = "http://localhost:11434/api/chat"
    model = "llama2"

    print(f"\n1. Testing connection to: {endpoint}")
    print(f"   Using model: {model}")

    # Simple test message
    test_data = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Say 'Hello, World!' and nothing else."}
        ],
        "stream": False
    }

    try:
        print(f"\n2. Sending test request...")
        response = requests.post(endpoint, json=test_data, timeout=30)

        print(f"   ✓ Status code: {response.status_code}")

        if response.status_code == 200:
            resp_json = response.json()
            print(f"   ✓ Response keys: {list(resp_json.keys())}")
            print(f"\n3. Full response:")
            print(json.dumps(resp_json, indent=2))

            if 'message' in resp_json and 'content' in resp_json['message']:
                print(f"\n4. ✓ Extracted content: {resp_json['message']['content']}")
                return True
            else:
                print(f"\n4. ✗ Unexpected response format!")
                return False
        else:
            print(f"   ✗ Error: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"   ✗ Connection failed!")
        print(f"   Is Ollama running? Try: ollama serve")
        return False
    except Exception as e:
        print(f"   ✗ Exception: {type(e).__name__}: {e}")
        return False


def test_json_generation():
    """Test if Ollama can generate valid JSON"""
    print("\n" + "="*60)
    print("JSON GENERATION TEST")
    print("="*60)

    endpoint = "http://localhost:11434/api/chat"
    model = "llama2"

    system_prompt = """You are a helpful assistant that responds ONLY in valid JSON format.
Your response must be a JSON object with exactly two keys: "improvement" and "prompt".
Example: {"improvement": "test improvement", "prompt": "test prompt"}"""

    user_prompt = """Generate a JSON response with these two fields:
1. "improvement": A suggestion to improve something
2. "prompt": A creative writing prompt

Respond ONLY with the JSON object, no other text."""

    test_data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 200
        }
    }

    try:
        print(f"\n1. Sending JSON generation request...")
        response = requests.post(endpoint, json=test_data, timeout=60)

        if response.status_code == 200:
            resp_json = response.json()
            content = resp_json.get('message', {}).get('content', '')

            print(f"\n2. Generated response:")
            print(content)

            # Try to extract JSON
            start = content.find("{")
            end = content.rfind("}") + 1

            if start != -1 and end > start:
                json_str = content[start:end]
                print(f"\n3. Extracted JSON: {json_str}")

                try:
                    import ast
                    parsed = ast.literal_eval(json_str)
                    print(f"\n4. ✓ Successfully parsed!")
                    print(f"   Keys: {list(parsed.keys())}")

                    if 'improvement' in parsed and 'prompt' in parsed:
                        print(f"   ✓ Has required keys!")
                        print(f"   improvement: {parsed['improvement']}")
                        print(f"   prompt: {parsed['prompt']}")
                        return True
                    else:
                        print(f"   ✗ Missing required keys!")
                        return False
                except Exception as e:
                    print(f"\n4. ✗ Failed to parse: {e}")
                    return False
            else:
                print(f"\n3. ✗ No JSON braces found in response!")
                return False
        else:
            print(f"   ✗ Error: {response.text}")
            return False

    except Exception as e:
        print(f"   ✗ Exception: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print("\n🔍 Starting Ollama diagnostics...\n")

    # Test 1: Basic connectivity
    test1_passed = test_ollama_connection()

    # Test 2: JSON generation (only if Test 1 passed)
    test2_passed = False
    if test1_passed:
        test2_passed = test_json_generation()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Test 1 (Connectivity): {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Test 2 (JSON Generation): {'✓ PASSED' if test2_passed else '✗ FAILED'}")

    if not test1_passed:
        print("\n⚠️  CRITICAL: Ollama is not responding!")
        print("   Fix: Run 'ollama serve' in another terminal")
        print("   Then: Run 'ollama pull llama2'")
    elif not test2_passed:
        print("\n⚠️  WARNING: Ollama works but JSON generation failed!")
        print("   The model may need better prompting or a different model")
        print("   Try: ollama pull mistral")
    else:
        print("\n✅ All tests passed! Ollama is ready.")

    print("="*60 + "\n")
