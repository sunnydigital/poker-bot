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
                model="gpt-4o-mini",
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
                                "text": "Analyze this poker table image and extract the game state."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "detail": "high"
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
            
            # Parse JSON string from response
            try:
                # Extract JSON part
                json_parts = raw_response.split("```")
                if len(json_parts) >= 2:
                    json_str = json_parts[1].strip()
                    if json_str.startswith("json"):
                        json_str = json_str[4:].strip()
                    
                    # Log extracted JSON string
                    logger.info(f"Extracted JSON string:\n{json_str}")
                    
                    result = json.loads(json_str)
                    
                    # Parse the inner boardState JSON string
                    if isinstance(result.get("boardState"), str):
                        board_state = json.loads(result["boardState"])
                        logger.info(f"Parsed board state:\n{json.dumps(board_state, indent=2)}")
                        return {"boardState": json.dumps(board_state)}
                    else:
                        logger.warning("boardState is not a string, returning as is")
                        return result
                else:
                    raise ValueError("No JSON found between ``` markers")
                    
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