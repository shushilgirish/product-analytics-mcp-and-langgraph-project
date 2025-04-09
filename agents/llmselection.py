import os
from typing import Any, Dict
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

class LLMSelector:
    """Class to manage LLM model selection and initialization."""
    
    # Available model mappings
    AVAILABLE_MODELS: Dict[str, str] = {
        "Claude 3 Haiku": "claude-3-haiku-20240307",     # Correct Anthropic model name
        "Claude 3 Sonnet": "claude-3-5-sonnet-20240620",   # Correct Anthropic model name
        "Gemini Pro": "gemini-2.0-flash",                      # Correct Google model name
        "DeepSeek": "deepseek-reasoner",      
        "Grok": "grok-2-latest"                              # Correct Grok model name
    }
    
    DEFAULT_MODEL = "Gemini Pro"

    def get_response(llm: Any, prompt: str) -> str:
        """Get response from LLM and extract content consistently."""
        try:
            response = llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"Error getting response: {str(e)}")
            return f"Error: {str(e)}"
    
    @staticmethod
    def get_llm(model_name: str = None) -> Any:
        """Initialize and return the appropriate LLM based on model name."""
        if not model_name:
            model_name = LLMSelector.DEFAULT_MODEL
        print(f"Initializing LLM with model: {model_name}")
            
        # Get the model ID from the friendly name
        model_id = LLMSelector.AVAILABLE_MODELS.get(model_name, 
                  LLMSelector.AVAILABLE_MODELS[LLMSelector.DEFAULT_MODEL])
            
        # Initialize based on model type
        if "claude" in model_id.lower():
            return ChatAnthropic(
                model=model_id,
                temperature=0,
                anthropic_api_key=os.environ.get('ANTHROPIC_API_KEY')
            )
        elif "gemini" in model_id.lower():
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_id,
                temperature=0,
                google_api_key=os.environ.get('GEMINI_API_KEY')
            )
        elif "deepseek" in model_id.lower():
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_id,
                temperature=0,
                api_key=os.environ.get('DEEP_SEEK_API_KEY')
            )
        elif "grok" in model_id.lower():
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=model_id,
                temperature=0,
                api_key=os.environ.get('GROK_API_KEY')
            )
        else:
            # Default to Claude 3 Haiku
            default_id = LLMSelector.AVAILABLE_MODELS[LLMSelector.DEFAULT_MODEL]
            return ChatAnthropic(
                model=default_id,
                temperature=0,
                anthropic_api_key=os.environ.get('ANTHROPIC_API_KEY')
            )

    @staticmethod
    def get_available_models() -> Dict[str, str]:
        """Return the list of available models."""
        return LLMSelector.AVAILABLE_MODELS.copy()

    # Update the count_tokens method
    @staticmethod
    def count_tokens(text: str, model_name: str) -> int:
        """Simple token estimation using model-specific ratios."""
        if not text:
            return 0
        
        # Simplified model ratios
        ratios = {"claude": 1.4, "gemini": 1.3, "grok": 1.3, "deepseek": 1.3}
        ratio = ratios.get(next((k for k in ratios if k in model_name.lower()), "gemini"))
        
        return int(len(text.split()) * ratio)    
    @staticmethod
    def get_token_limits(model_name: str) -> dict:
        """Get context window and token limits for each model."""
        return {
            "Claude 3 Haiku": {"context_window": 200000, "cost_per_1k": 0.25},
            "Claude 3 Sonnet": {"context_window": 200000, "cost_per_1k": 0.50},
            "Gemini Pro": {"context_window": 32768, "cost_per_1k": 0.10},
            "DeepSeek": {"context_window": 32768, "cost_per_1k": 0.25},
            "Grok": {"context_window": 32768, "cost_per_1k": 0.25}
        }.get(model_name, {"context_window": 32768, "cost_per_1k": 0.25})



if __name__ == "__main__":
    print("Testing LLM Selection...")
    
    test_models = [
        "Gemini Pro"
    ]
    
    for model in test_models:
        print(f"\nTesting model: {model}")
        try:
            llm = LLMSelector.get_llm(model)
            test_prompt = "Hello, can you hear me?"
            response = llm.invoke(test_prompt)
            # Access the content of the AIMessage
            content = response.content if hasattr(response, 'content') else str(response)
            print(f"Response received: {content[:100]}...")
        except Exception as e:
            print(f"Error testing {model}: {str(e)}")