import os
import json
from openai import OpenAI
from typing import Dict, Any
from dotenv import load_dotenv

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
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Please set it in .env file or pass it directly.")
        self.client = OpenAI(api_key=self.api_key)

    def analyze_poker_image(self, image_base64: str) -> Dict[str, Any]:
        """
        Analyze poker table image
        Args:
            image_base64: base64 encoded image
        Returns:
            Dict: Dictionary containing game state
        """
        try:
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
                                "type": "image",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            # Parse JSON string from response
            result = response.choices[0].message.content
            # Extract JSON part
            json_str = result.split("```")[1].strip()
            if json_str.startswith("json"):
                json_str = json_str[4:].strip()
            
            return json.loads(json_str)
            
        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            return {
                "boardState": json.dumps({
                    "isGameOver": False,
                    "communityCards": [],
                    "holeCards": [],
                    "currentPotValue": 0,
                    "whoRaised": []
                })
            } 