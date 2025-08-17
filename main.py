from langchain.chat_models import init_chat_model
# from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
# from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from tools import create_wiki_tool, create_arxiv_tool, create_web_page_tool, create_search_tool

import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage

def convert_to_lc_messages(messages):
    lc_messages = []
    for m in messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))
    return lc_messages





def main():

    st.title("🔎 LangChain - Chat with search")
    """
    In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
    Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
    """

    ## Sidebar for settings
    st.sidebar.title("Settings")
    api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")
    web_page_url=st.sidebar.text_input("Enter Any web page URL:",type="default",value="https://docs.smith.langchain.com/")

    if "messages" not in st.session_state:
        st.session_state["messages"]=[
            {"role":"assistant","content":"Hi,I'm a chatbot who can search the web. How can I help you?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg['content'])

    if prompt:=st.chat_input(placeholder="What is machine learning?"):
        st.session_state.messages.append({"role":"user","content":prompt})
        st.chat_message("user").write(prompt)

        llm=init_chat_model("Llama3-8b-8192", model_provider="groq", max_tokens = 1000, api_key=api_key, streaming=True)
        tools=[create_arxiv_tool(),create_search_tool(),create_wiki_tool(),create_web_page_tool(web_page_url)]

        agent_executor = create_react_agent(llm, tools)

        with st.chat_message("assistant"):
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            # response=agent_executor.run(st.session_state.messages,callbacks=[st_cb])

            # input_message = {"role": "user", "content": "What is LangSmith? tell me pointwise adwantages. provide 5 points"}
            lc_messages = convert_to_lc_messages(st.session_state.messages)

            response = agent_executor.invoke({"messages": lc_messages}, callbacks=[st_cb])

            # The response is an AIMessage (or dict with output), so get text safely:
            answer = response["messages"][-1].content if "messages" in response else str(response)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.write(answer)





if __name__ == "__main__":
    main()