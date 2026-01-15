from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END

# We will use this model for both the conversation and the summarization
from langchain_aws import ChatBedrockConverse
model = ChatBedrockConverse(
    model="us.meta.llama4-scout-17b-instruct-v1:0",
    region_name="us-east-1",
    temperature=0.5,
    max_tokens=2048,
    top_p=0.9,
)

# State class to store messages and summary
class State(MessagesState):
    summary: str
    
# Define the logic to call the model
def call_model(state: State):
    
    # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, then we add it to messages
    if summary:
        
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]
    
    else:
        messages = state["messages"]
    
    # Ensure messages start with a HumanMessage (required by Bedrock Converse API)
    # Filter out any leading AIMessages before the first HumanMessage
    filtered_messages = []
    found_human = False
    for msg in messages:
        if isinstance(msg, SystemMessage):
            filtered_messages.append(msg)
        elif isinstance(msg, HumanMessage):
            found_human = True
            filtered_messages.append(msg)
        elif found_human:
            # Only include AIMessages after we've seen a HumanMessage
            filtered_messages.append(msg)
    
    # Ensure the last message is a HumanMessage (also required by Llama models on Bedrock)
    # Remove trailing AIMessages
    while filtered_messages and isinstance(filtered_messages[-1], AIMessage):
        filtered_messages.pop()
    
    # If no messages left or no HumanMessage, add a placeholder
    if not filtered_messages or not any(isinstance(m, HumanMessage) for m in filtered_messages):
        filtered_messages = [HumanMessage(content="Hello")]
    
    response = model.invoke(filtered_messages)
    return {"messages": response}

# Determine whether to end or summarize the conversation
def should_continue(state: State) -> Literal["summarize_conversation", "__end__"]:
    
    """Return the next node to execute."""
    
    messages = state["messages"]
    
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    
    # Otherwise we can just end
    return END

def summarize_conversation(state: State):
    
    # First get the summary if it exists
    summary = state.get("summary", "")

    # Create our summarization prompt 
    if summary:
        
        # If a summary already exists, add it to the prompt
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        # If no summary exists, just create a new one
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    
    # Ensure messages start with a HumanMessage (required by Bedrock Converse API)
    filtered_messages = []
    found_human = False
    for msg in messages:
        if isinstance(msg, SystemMessage):
            filtered_messages.append(msg)
        elif isinstance(msg, HumanMessage):
            found_human = True
            filtered_messages.append(msg)
        elif found_human:
            filtered_messages.append(msg)
    
    # Ensure the last message is a HumanMessage (required by Llama models on Bedrock)
    while filtered_messages and isinstance(filtered_messages[-1], AIMessage):
        filtered_messages.pop()
    
    # If no HumanMessage, add the summary request as the only message
    if not filtered_messages or not any(isinstance(m, HumanMessage) for m in filtered_messages):
        filtered_messages = [HumanMessage(content=summary_message)]
    
    response = model.invoke(filtered_messages)
    
    # Delete all but the 3 most recent messages to ensure a HumanMessage is preserved
    # (Bedrock Converse requires conversations to start with a user message)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-3]]
    return {"summary": response.content, "messages": delete_messages}

# Define a new graph
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

# Compile
graph = workflow.compile()