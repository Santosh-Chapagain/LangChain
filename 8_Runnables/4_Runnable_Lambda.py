from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough , RunnableLambda
import os
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)
model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="Write small joke on \n {topic}",
    input_variables=['topic']

)

parser = StrOutputParser()
def word_count(joke):
    return len(joke.split())


joke_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'length': RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_chain, parallel_chain)

result = final_chain.invoke({'topic': 'AI'})

print(result)
