import json

import streamlit as st
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser
from langchain.text_splitter import CharacterTextSplitter


class JsonOutputParser(BaseOutputParser):
    def parse(self, text:str):
        text = text.replace("```","").replace("json","")
        return json.loads(text)

output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGpt",
    page_icon="🤔",
)

st.title("Quiz Gpt")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks= [StreamingStdOutCallbackHandler()],
)

def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
question_prompt = ChatPromptTemplate.from_messages([
        ("system","""
        You are a helpful assistant that is role playing as a teacher.
        
        Based ONLY on the following context make 10 question to test the user's knowledge about the text.
        
        Each question should have 4 answers, three of them must be incorrect and one should be correct.
        
        Use (o) to signal the correct answer.
        
        Question example:
        
        Question: what is the color of ocean?
        Answers: Red| Yellow| Green| Blue(o)
        
        Question: What is the capital or Georgia?
        Answers: Baku|Tbilisi(o)|Manila|Beirut
        
        Question: When was Avatar released?
        Answers: 2007|2001|2009(o)|1998
        
        Question: Who was Julius Caesar?
        Answers: A Roman Emperor(o)|Painter|Actor|Model
        
        Your turn!
        
        Context: {context}
        """),
        
    ])

formatting_prompt = ChatPromptTemplate.from_messages([
    (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }},
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }},
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }},
                ]
            }}
        ]
     }}
    ```
    Your turn!
    Questions: {context}
""",
        )
])

formatting_chain = formatting_prompt | llm

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path,"wb") as f:
        f.write(file_content)
    
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator = "\n",
        chunk_size = 600,
        chunk_overlap = 100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

question_chain = {
        "context":format_docs
        }|question_prompt | llm
    
@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs,topic):
    chain = {"context": question_chain}| formatting_chain |output_parser
    return chain.invoke(_docs)

@st.cache_data(show_spinner="Searching Wiki...")
def wiki_search_chain(term):
    retriever = WikipediaRetriever(top_k_results=2)
    docs = retriever.get_relevant_documents(topic)
    return docs

with st.sidebar:
    docs = None
    choice = st.selectbox("Choose what you want to use",("File","Wikipedia Article"),)
    if choice == "File":
        topic = False
        file = st.file_uploader("Upload a .docx, .txt, .pdf file",type=["pdf","txt","docx"],)
        if file :
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedea...")
        if topic:
            docs = wiki_search_chain(topic)
            
if not docs:
    st.markdown("""
    Welcome to Quiz GPT,
    
    I will make a quiz from Wikipedia or file you upload!
    
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """)
else :
    
    
    
    

    response = run_quiz_chain(docs,topic if topic else file.name)
    # st.write(response)
    with st.form("questions_form"):
        for question in response["questions"]:
            st.subheader(question["question"])
            value=  st.radio("Select an option.",[answer["answer"] for answer in question["answers"]],)
            if ({"answer":value,"correct":True} in question["answers"] ):
                st.success("Correct!")
            elif value is not None:
                st.error("Wrong")
            
                
        st.form_submit_button()
