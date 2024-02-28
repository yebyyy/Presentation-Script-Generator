import os
import Api_key
import streamlit as st
from langchain_openai import OpenAI
import wikipedia
import chromadb
import tiktoken
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import WikipediaAPIWrapper
os.environ["OPENAI_API_KEY"] = Api_key.apikey

# Create an instance of the presentation script generator
st.title("✏️Presentation Script Generator")

# Create a text input for the user to enter a prompt
prompt = st.text_input("Enter a prompt for the AI to generate a presentation script")

title_template = PromptTemplate(
    input_variables=["topic"],
    template="Create a presentation argument or title of {topic}."
)

# #I want the user to input a numerical value to represent the number of minutes the presentation should be
# minutes = st.number_input("How many minutes long do you want the presentation to be?")

script_template = PromptTemplate(
    input_variables=["title"],
    template="Create a presentation script that is about {title}."
)

#Memory
title_memory = ConversationBufferMemory(input_key="topic", memory_key="chat_history")
script_memory = ConversationBufferMemory(input_key="title", memory_key="chat_history")

# # Calculate the number of sentences needed based on the minutes input
# num_sentences = int(minutes * 150)  # Assuming an average of 150 words per minute

# # Generate the presentation script with the desired number of sentences
# script = sequence_chain.run(prompt, max_tokens=num_sentences)

# # Display the generated script
# st.write(script)


llms = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llms, prompt=title_template, output_key="title", memory=title_memory)
script_chain = LLMChain(llm=llms, prompt=script_template, output_key="script", memory=script_memory)


if prompt:
    # Generate a presentation script using the prompt
    title = title_chain.run(prompt)
    script = script_chain.run(title)
    st.write(title)
    st.write(script)

    with st.expander("Title History"):
        st.info(title_memory.buffer)

    with st.expander("Script History"):
        st.info(script_memory.buffer)