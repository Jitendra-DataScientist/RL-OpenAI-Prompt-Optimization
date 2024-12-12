"""
same as trial1.py except that the reward is being used by user (perhaps Shashank)
"""

import os
from dotenv import load_dotenv
import random
from openai import OpenAI
import json

# Load the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
# print(openai_api_key)

client = OpenAI(api_key=openai_api_key)

# Load environment variables from a .env file
load_dotenv()

if not openai_api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

# Set the API key for OpenAI


def get_response(prompt):
    """
    Interacts with the OpenAI API using the given prompt.
    """
    try:
        response = client.completions.create(model="gpt-3.5-turbo-instruct",  # Use the appropriate model
        prompt=prompt,
        max_tokens=50)
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error: {e}")
        return None

def reward_function(response):
    """
    Evaluates the response and returns a reward score.
    Simple example: Longer responses get higher rewards.
    """
    # if not response:
    #     return -1  # Penalize if no response
    # return len(response)  # Reward based on response length
    reward = input("Enter the reward:")
    return float(reward)

def optimize_prompt(initial_prompt, iterations=10):
    """
    Uses reinforcement learning to optimize the prompt.
    """
    prompt = initial_prompt
    learning_rate = 0.1  # How much to adjust the prompt
    history = []

    for i in range(iterations):
        print(f"Iteration {i+1}")

        # Get a response from the OpenAI API
        response = get_response(prompt)
        print(f"Response: {response}")

        # Calculate reward
        reward = reward_function(response)
        print(f"Reward: {reward}")

        # Update prompt (simple gradient ascent)
        # In practice, you may use more sophisticated methods
        adjustment = random.choice([" add details", " clarify", " explain further"])
        if reward > 0:
            prompt += adjustment
        else:
            prompt = initial_prompt  # Reset if response is bad

        history.append((prompt, reward))

    return prompt, history

# Example usage
# initial_prompt = "Tell me about AI."
initial_prompt = input("Enter initial prompt:")
final_prompt, training_history = optimize_prompt(initial_prompt, iterations=3)

print("\nFinal Optimized Prompt:", final_prompt)
print("Training History:")
print (json.dumps(training_history,indent=4))
