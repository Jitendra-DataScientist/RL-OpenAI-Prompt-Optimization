"""
Explanation:
Initialization:

The initial_prompt is the starting point for optimization.
iterations specify how many times the prompt is updated.
Interaction:

The get_response function sends the prompt to the OpenAI API and retrieves a response.
Reward Function:

The reward_function evaluates the quality of the response. Here, it uses the response length as a proxy for quality.
Optimization:

The optimize_prompt function applies simple gradient ascent by adding adjustments to the prompt based on the reward.
This example uses random text adjustments, but you can integrate more structured updates based on NLP techniques.
Output:

The final optimized prompt and training history are returned for review.
Next Steps:
Improve the reward function to better capture the response's relevance or correctness.
Use advanced RL algorithms, such as Proximal Policy Optimization (PPO), for better results.
Fine-tune the model parameters for specific tasks if you have access to a custom API environment.
"""

import os
from dotenv import load_dotenv
import random
from openai import OpenAI

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
    if not response:
        return -1  # Penalize if no response
    return len(response)  # Reward based on response length

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
initial_prompt = "Tell me about AI."
final_prompt, training_history = optimize_prompt(initial_prompt, iterations=5)

print("\nFinal Optimized Prompt:", final_prompt)
print("Training History:", training_history)
