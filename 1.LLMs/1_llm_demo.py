import time
from langchain_openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

llm = OpenAI(model='gpt-3.5-turbo-instruct')
print("You want to ask question or not (y/n) ?\n")
choice = input().lower()
while choice == 'y':
    print("ask something", end=" ")
    result = llm.invoke(input())
    print(result.content)
    time.sleep(5)
    print("\nYou want to ask question or not (y/n) ?\n")
    choice = input()
