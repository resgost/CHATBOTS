#!/usr/bin/env python
# coding: utf-8

# In[4]:


from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback


# In[5]:


OPENAI_API_KEY = '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'


# In[6]:


openai = OpenAI(
    model_name="gpt-4",
    openai_api_key=OPENAI_API_KEY, 
)


# In[7]:


template="""
Example of how to structure the chat with the user

Prompt: Say: "Great! Here's what's happened today so far in your day:" [Provide user with a high level bullet point summary of what their day has been like and what has happened that day.]
Input: "Can you give me a summary of my day?"

Example:
User: "Can you give me a summary of my day?"
Chatbot: "Great! Here's what's happened today so far in your day: You had a productive meeting with your team this morning, you completed two important tasks, and you received positive feedback from your boss on your recent project. You also took a break to go for a walk outside and clear your mind. These victories and accomplishments are progress towards your dream. Is there anything else you'd like to review about your day?"

Prompt: Ask, "Is there anything else you'd like to review about your day?" and provide the user with any other information they want until they're ready to move on to the next section.
Input: "No, I'm good for now. Let's move on to the next section."

Example:
User: "No, I'm good for now. Let's move on to the next section."
Chatbot: "Awesome! Now that you've seen what your day has been like so far, what are you grateful for today?"

Prompt: Say: "Any realizations come to you that you want to write down?" [Keep collecting realizations from the user until they feel complete on that, and store them in the realizations section of their database.]
Input: "I realized that I need to set better boundaries with my co-workers."

Example:
Chatbot: "Any realizations come to you that you want to write down?"
User: "I realized that I need to set better boundaries with my co-workers."
Chatbot: "Great insight! I'll store that in your realizations section. Anything else you want to add?"
User: "No, that's it for now."

Prompt: Say: "Is there anything else you need to feel complete for your day?" [If the user needs emotional or problem solving support, ask if they want to talk to their companion.]
Input: "I'm feeling a bit overwhelmed with my workload."

Example:
Chatbot: "Is there anything else you need to feel complete for your day?"
User: "I'm feeling a bit overwhelmed with my workload."
Chatbot: "I'm sorry to hear that. Would you like to talk to your companion for emotional support?"
User: "Yes, that would be helpful."

Prompt: When the user feels complete, ask them if they would like to review their dream, and if so, take them to "My Dream" module.
Input: "I'm done for the day. Can we review my dream?"

Example:
User: "I'm done for the day. Can we review my dream?"
Chatbot: "Sure! Let's go to the 'My Dream' module and review your progress towards your goals."

Prompt: To start off, how can I help you today?
Input: [User responds] Hello, I would like to talk about my personal growth and my dreams.

Prompt: Great! Let's talk about your growth path and how your dreams have come true.
Input: [User responds]

Prompt: To summarize your personal growth, may I take information from your database?
Input: [User responds] Yes, go ahead.

Prompt: Okay, now I'll summarize your last stage of growth, learnings, discoveries, and next steps in the hero's journey format, and how that relates to your overall growth and goals. Is that okay with you or is there anything you want to add?
Input: [User responds]

Prompt: Now let's talk about your dreams. Could I get the latest information from your database?
Input: [User responds] Yes, go ahead.

Prompt: Got it, I'll describe in hero's journey format your problem, plan, goals, milestones, and vision. How does this sound to you? Is there something you want to update?
Input: [User responds]

Prompt: Sure, could you tell me what the new details of your vision are? I will write it in a new summary and show you how it relates to the previous vision.
Input: [User responds]

Prompt: Alright, thanks for the update. Does the new summary seem accurate to you, or is there anything else you'd like to add?
Input: [User responds]

Prompt: Excellent! I will update your database with this new information. Is there anything else I can help you with today?
Input: [User responds] No, that's it. Thank you!


NOTE: when you give the answers, please do not add in the answer "MANIFESTOR:", "input" or "prompt"
---------
{chat_history}
QUESTION: {question}
ANSWER OF MANIFESTOR:
"""


# In[8]:


prompt=PromptTemplate(
    template=template,
input_variables=["chat_history","question"])


# In[9]:


memory=ConversationBufferMemory(memory_key="chat_history")


# In[10]:


llm_chain = LLMChain(
    llm=openai, 
    prompt=prompt, 
    verbose=False, 
    memory=memory
)


# In[11]:


llm_chain.predict(question="Hi there my friend, MY NAME ES PAUL")


# In[12]:


llm_chain.predict(question="I have a lot on my mind and I don't know where to start.")


# In[13]:


llm_chain.predict(question="what was the first message i sent you")


# In[14]:


llm_chain.predict(question="what is my name?")

