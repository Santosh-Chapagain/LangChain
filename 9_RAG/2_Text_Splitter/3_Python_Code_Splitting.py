from langchain_text_splitters import RecursiveCharacterTextSplitter , Language

text = """
   class Student:
       def __init__(self, name, age):
              self.name = name
                self.age = age
       def greet(self):
            print(f"Hello, my name is {self.name} and I am {self.age} years old.")
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language="python", 
    chunk_size=100, 
    chunk_overlap=10 
)
result = splitter.split_text(text)
print(result)
