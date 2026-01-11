from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()


template = PromptTemplate(
    template= 'Write detailed summary on {topic}' ,
    input_variables=['topic']

)


parser = StrOutputParser()

template2 = PromptTemplate(
    template='Write major 5 point from {report}',
    input_variables=['report']
)


model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

chain = template | model | parser | template2 | model | parser 

result = chain.invoke({'topic': 'Antimatter'})


print(result)
chain.get_graph().print_ascii()
             
