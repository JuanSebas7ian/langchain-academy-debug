
import httpx
import time

def check_server_is_running(url="http://127.0.0.1:2024", retries=3):
    """
    Checks if the LangGraph server is running at the given URL.
    """
    print(f"Checking if LangGraph server is running at {url}...")
    try:
        # Check specific endpoint that should be available
        response = httpx.get(f"{url}/docs", timeout=2.0)
        if response.status_code == 200:
            print("✅ LangGraph server is running and accessible.")
            return True
    except (httpx.ConnectError, httpx.TimeoutException):
        pass
    
    print(f"❌ Could not connect to {url}.")
    print("⚠️  Make sure you have started the LangGraph server in a separate terminal:")
    print("   cd module-1/studio")
    print("   langgraph dev")
    return False
