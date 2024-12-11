# Poker Bot with LLM Vision

An intelligent poker bot that uses LLM (Large Language Model) vision capabilities to analyze poker game screenshots and make strategic decisions.

## Features

- **Automatic Game State Recognition**: Uses GPT-4 Vision to analyze poker table screenshots
- **Intelligent Decision Making**: LLM-based poker strategy
- **Automated Actions**: Automatically executes poker actions (fold, call, check, raise)
- **Button Detection**: Automatic detection of poker interface buttons using template matching
- **Flexible Raise Control**: Precise control of raise amounts using OCR and button automation

## Prerequisites

- Python 3.8+
- OpenAI API key
- Tesseract OCR

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd poker-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:
- Windows: Download from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- Linux: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Configuration

1. Place button template images in the `assets` folder:
- `fold_button.png`
- `call_button.png`
- `check_button.png`
- `raise_to_button.png`
- `min_button.png`
- `max_button.png`
- `plus_button.png`
- `minus_button.png`
- `pot_button.png`
- `fifty_percent_button.png`

2. Adjust OCR and button detection parameters in `utils/action_executor.py` if needed.

## Usage

1. Start the bot:
```bash
python main.py
```

2. The bot will:
- Initialize by locating all necessary buttons
- Continuously monitor the game screen
- Analyze the game state using LLM vision
- Make strategic decisions
- Execute actions automatically

3. To stop the bot, press `Ctrl+C`

## Project Structure

```
poker-bot/
├── main.py              # Main program entry point
├── game_state.py        # Game state management
├── poker_agent.py       # AI decision making
├── utils/
│   ├── action_executor.py   # Action execution
│   ├── llm_api.py          # LLM API integration
│   └── screen_capture.py    # Screen capture
├── assets/              # Button template images
└── requirements.txt     # Project dependencies
```

## Safety Features

- Mouse to top-left corner will abort the program (PyAutoGUI failsafe)
- Delay between actions to prevent rapid clicking
- Error handling and logging
- Configurable confidence thresholds for button detection

## Limitations

- Requires clear visibility of the poker interface
- Button templates must match the actual poker client interface
- OCR accuracy depends on the clarity of displayed numbers
- Requires stable internet connection for LLM API calls

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This bot is for educational purposes only. Using automated bots in real money poker games may violate terms of service and could be illegal in some jurisdictions. Use responsibly and at your own risk.
