import os
import json
from openai import OpenAI
from typing import Dict, Any
from dotenv import load_dotenv
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"llm_analyzer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

POKER_SYSTEM_PROMPT = """
Analyze the provided poker table image and extract the current game state information.

Expected response format:

```
boardState: A JSON string containing game parameters, for example:
 "{
    "isGameOver": false,
    "communityCards": ["A-S", "2-H", "3-C"], 
    "holeCards": ["5-D", "6-S"], 
    "currentPotValue": 64,
    "whoRaised": [
        {
            "name": "John",
            "raise": 20,
        }
    ]
}"
```

Valid card values include: 
{'2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'}

Valid card suits include: 
{'H', 'D', 'C', 'S'}

Response must be formatted as:

```
{ "boardState": "..." }
```

Note: communityCards refers to shared cards visible on the table, while holeCards represents your private cards. You must identify exactly 2 holeCards in every response.
"""

class LLMAnalyzer:
    def __init__(self, api_key: str = None):
        """
        Initialize LLM analyzer
        Args:
            api_key: OpenAI API key, if None will get from environment variables
        """
        self.api_key = api_key or os.getenv("GPT_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Please set it in .env file or pass it directly.")
        self.client = OpenAI(api_key=self.api_key)
        logger.info("Initialized LLM Analyzer")

    def analyze_poker_image(self, image_base64: str) -> Dict[str, Any]:
        """
        Analyze poker table image
        Args:
            image_base64: base64 encoded image
        Returns:
            Dict: Dictionary containing game state
        """
        try:
            logger.info("Starting poker image analysis")
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": POKER_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this poker table image and extract the game state. Return the response in a single line without any formatting."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            # Log raw response
            raw_response = response.choices[0].message.content
            logger.info(f"Raw LLM response:\n{raw_response}")
            
            try:
                # Remove code block markers and clean up the string
                clean_response = raw_response.replace('```', '').strip()
                if clean_response.startswith('json'):
                    clean_response = clean_response[4:].strip()
                
                # Remove all whitespace and newlines
                clean_response = ''.join(clean_response.split())
                
                # Log cleaned response
                logger.info(f"Cleaned response:\n{clean_response}")
                
                # Parse outer JSON
                outer_json = json.loads(clean_response)
                
                # Get boardState string and clean it
                board_state_str = outer_json.get("boardState", "{}")
                
                # Remove any surrounding quotes and unescape
                if board_state_str.startswith('"') and board_state_str.endswith('"'):
                    board_state_str = board_state_str[1:-1]
                board_state_str = board_state_str.replace('\\"', '"').replace('\\\\', '\\')
                
                # Remove any trailing quotes
                board_state_str = board_state_str.rstrip('"')
                
                # Remove all whitespace and newlines
                board_state_str = ''.join(board_state_str.split())
                
                # Log cleaned boardState
                logger.info(f"Cleaned boardState string:\n{board_state_str}")
                
                try:
                    # Parse boardState
                    board_state = json.loads(board_state_str)
                    
                    # Validate the parsed state
                    required_fields = ["isGameOver", "communityCards", "holeCards", "currentPotValue", "whoRaised"]
                    if not all(field in board_state for field in required_fields):
                        missing_fields = [field for field in required_fields if field not in board_state]
                        logger.error(f"Missing required fields in board state: {missing_fields}")
                        raise ValueError(f"Missing required fields: {missing_fields}")
                    
                    # Format the output nicely for logging
                    formatted_state = json.dumps(board_state, indent=2)
                    logger.info(f"Successfully parsed board state:\n{formatted_state}")
                    
                    return {"boardState": json.dumps(board_state)}
                    
                except json.JSONDecodeError as je:
                    logger.error(f"Error parsing boardState JSON: {str(je)}")
                    logger.error(f"Problematic boardState string:\n{board_state_str}")
                    raise
                
            except Exception as parse_error:
                logger.error(f"Error parsing response: {str(parse_error)}")
                logger.error(f"Problematic response:\n{raw_response}")
                raise
            
        except Exception as e:
            error_msg = f"Error analyzing image: {str(e)}"
            logger.error(error_msg)
            default_state = {
                "isGameOver": False,
                "communityCards": [],
                "holeCards": [],
                "currentPotValue": 0,
                "whoRaised": []
            }
            logger.info(f"Returning default state: {json.dumps(default_state, indent=2)}")
            return {"boardState": json.dumps(default_state)} 