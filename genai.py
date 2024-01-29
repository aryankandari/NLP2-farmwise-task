# from langchain.chains import LLMChain
# from langchain_community.llms import OpenAI
# from langchain_core.prompts import PromptTemplate

# api_key = "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"
# llm = OpenAI(api_key=api_key)

# prompt_template = "Question: {question}\nAnswer: Let's think step by step."
# prompt = PromptTemplate(
#     input_variables=["question"], template=prompt_template
# )

# llm_chain = LLMChain(llm=llm, prompt=prompt)

# inputs = {"question": "What is the capital of France?"}
# output = llm_chain(inputs)
# print(output)

from collections import UserString
from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


speech_file_path = Path(__file__).parent / "speech.mp3"
response = client.audio.speech.create(
  model="tts-1",
  voice="alloy",
  input="Hi my name is Aryan Kandari, a final year B.tech undergrad today I welcome you to my project of topic recommendation using generative AI, i would like to thank you all for visiting my project and do let me know what you liked the most and what else I can improve in my project."
)

response.stream_to_file(speech_file_path)