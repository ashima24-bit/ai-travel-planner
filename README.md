# AI Travel Planner

A command-line **AI Travel Planner** built with **LangChain** that generates personalized travel itineraries using Prompt Templates, Chains, Memory, and Tool Calling.

Runs in **Mock Mode** out of the box — no API keys required. For real AI responses, plug in a free Groq API key (see [Using Real AI](#using-real-ai-optional)).

## Features

- **Prompt Templates** — Structured prompts to collect travel details
- **Sequential Chains** — End-to-end pipeline: Details → Itinerary → Budget → Tips
- **Conversation Memory** — Modify trips without re-entering details
- **Tool Calling** — Weather, Currency Conversion, and Attractions tools
- **Export** — Save itinerary to a text file
- **Mock Mode** — Works offline without API keys

## Requirements

- Python 3.10+
- pip

## Setup

```bash
# 1. Navigate to project
cd ai-travel-planner

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

Follow the prompts:

1. Enter destination, days, budget, interests, and dates
2. View your personalized itinerary, budget breakdown, and travel tips
3. Use the options menu to:
   - Modify trip (change budget, days, or destination)
   - Export itinerary to a text file
   - Convert currency
   - Check weather
   - Find nearby attractions & restaurants

## Sample Input

```
Destination  : Manali
Days         : 4
Budget (₹)   : 20000
Interests    : Adventure, Nature
Dates        : Dec 15-18
```

## Project Structure

```
ai-travel-planner/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── README.md           # Documentation
└── exports/            # Exported itineraries
```

## Using Real AI (Optional)

1. Get a free API key from [console.groq.com](https://console.groq.com) (no credit card needed)
2. Install: `pip install langchain-groq`
3. Set key:
   - Windows: `set GROQ_API_KEY=your_key_here`
   - Mac/Linux: `export GROQ_API_KEY=your_key_here`
4. In `app.py`, change `mock_mode=True` to `mock_mode=False`

## License

This project is for educational purposes.
