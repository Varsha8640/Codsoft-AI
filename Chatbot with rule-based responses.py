def chatbot_response(user_input):
  """
  This function takes user input and returns a response based on simple rules.

  Args:
      user_input: A string containing the user's message.

  Returns:
      A string containing the chatbot's response.
  """

  # Define simple rules and responses
  greetings = ["hello", "hi", "hey"]
  goodbye = ["bye", "goodbye", "see you later"]
  weather_questions = ["weather", "how's the weather"]

  # Check for greetings
  if any(word in user_input.lower() for word in greetings):
    return "Hello! How can I help you today?"

  # Check for goodbyes
  if any(word in user_input.lower() for word in goodbye):
    return "Goodbye! Have a nice day."

  # Check for weather questions
  if any(word in user_input.lower() for word in weather_questions):
    return "I'm sorry, I don't have access to real-time weather information."

  # Default response for other inputs
  return "I'm not sure I understand. Can you rephrase your question?"

# Get user input
user_input = input("You: ")

# Get and print the chatbot's response
response = chatbot_response(user_input)
print("Chatbot: ", response)