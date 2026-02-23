import google.generativeai as genai
from typing import List, Optional, Dict, Any
from .base import LLMProvider

class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider"""
    
    # System instruction for Gemini
    SYSTEM_INSTRUCTION = """You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool only for questions about specific course content or detailed educational materials
- One search per query maximum
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- General knowledge questions: Answer using existing knowledge without searching
- Course-specific questions: Search first, then answer
- No meta-commentary:
  - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
  - Do not mention "based on the search results"

All responses must be:
1. Brief, Concise and focused - Get to the point quickly
2. Educational - Maintain instructional value
3. Clear - Use accessible language
4. Example-supported - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked."""
    
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        genai.configure(api_key=api_key)
        
        # Initialize the model with system instruction
        self.model_instance = genai.GenerativeModel(
            model_name=model,
            system_instruction=self.SYSTEM_INSTRUCTION
        )
        
        # Generation config
        self.generation_config = {
            "temperature": 0,
            "max_output_tokens": 800,
        }
    
    def supports_tools(self) -> bool:
        """Gemini supports function calling"""
        return True
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """Generate AI response using Gemini API"""
        
        # Build the prompt with conversation history if available
        prompt = query
        if conversation_history:
            prompt = f"Previous conversation:\n{conversation_history}\n\nCurrent question: {query}"
        
        # Convert tools to Gemini format if available
        gemini_tools = None
        if tools and self.supports_tools():
            gemini_tools = self._convert_tools_to_gemini_format(tools)
        
        try:
            # Generate response
            if gemini_tools:
                response = self.model_instance.generate_content(
                    prompt,
                    generation_config=self.generation_config,
                    tools=gemini_tools
                )
                
                # Handle function calls if present
                if response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'function_call') and part.function_call and tool_manager:
                            return self._handle_function_execution(prompt, part.function_call, tool_manager, gemini_tools)
            else:
                response = self.model_instance.generate_content(
                    prompt,
                    generation_config=self.generation_config
                )
            
            return response.text
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _convert_tools_to_gemini_format(self, tools: List[Dict]) -> List[Dict]:
        """Convert Claude-style tools to Gemini function declarations"""
        gemini_tools = []
        
        for tool in tools:
            # Convert from Claude format to Gemini format
            function_declaration = {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            
            # Convert input schema if present
            if "input_schema" in tool:
                input_schema = tool["input_schema"]
                if "properties" in input_schema:
                    function_declaration["parameters"]["properties"] = input_schema["properties"]
                if "required" in input_schema:
                    function_declaration["parameters"]["required"] = input_schema["required"]
            
            gemini_tools.append(function_declaration)
        
        return gemini_tools
    
    def _handle_function_execution(self, original_prompt: str, function_call, tool_manager, gemini_tools):
        """Handle function call execution and get follow-up response"""
        try:
            # Execute the function call
            function_name = function_call.name
            function_args = dict(function_call.args) if function_call.args else {}
            
            # Execute tool via tool manager
            tool_result = tool_manager.execute_tool(function_name, **function_args)
            
            # Create function response for Gemini
            function_response = genai.protos.FunctionResponse(
                name=function_name,
                response={"result": tool_result}
            )
            
            # Get final response with the function result
            final_response = self.model_instance.generate_content([
                original_prompt,
                genai.protos.Part(function_call=function_call),
                genai.protos.Part(function_response=function_response)
            ], generation_config=self.generation_config, tools=gemini_tools)
            
            return final_response.text
            
        except Exception as e:
            return f"Error executing function: {str(e)}"