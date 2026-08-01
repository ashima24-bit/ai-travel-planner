"""
AI Travel Planner using LangChain
Mini Project - Prompt Templates, Chains, Memory, Tools & Tool Calling

Runs in MOCK mode by default (no API keys needed).
For real AI, get a free Groq API key at https://console.groq.com
"""

import os
import re
import sys
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# Fix Unicode display on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda


# =========================================================================
# MOCK LLM - Works without any API keys
# =========================================================================

class MockLLM:
    """Mock LLM that generates realistic travel responses.
    Implements enough of the LangChain interface to work with chains."""

    temperature: float = 0.7

    def __init__(self):
        self._memory: Dict[str, str] = {}

    def predict(self, prompt: str) -> str:
        return self._route_prompt(prompt)

    def generate(self, prompts: List[str]) -> List[str]:
        return [self._route_prompt(p) for p in prompts]

    def invoke(self, messages):
        """Handle both single messages and lists."""
        if isinstance(messages, list):
            text = "\n".join(
                m.content if hasattr(m, 'content') else str(m)
                for m in messages
            )
        else:
            text = str(messages)
        return AIMessage(content=self._route_prompt(text))

    def _route_prompt(self, prompt: str) -> str:
        pl = prompt.lower()
        if "weather" in pl and "itinerary" not in pl:
            return self._weather_response(prompt)
        if "currency" in pl or "convert" in pl:
            return self._currency_response(prompt)
        if ("budget" in pl and "breakdown" in pl) or "budget estimate" in pl:
            return self._budget_response(prompt)
        if "packing" in pl or "travel tip" in pl:
            return self._tips_response(prompt)
        if "itinerary" in pl or ("day" in pl and "budget" not in pl):
            return self._itinerary_response(prompt)
        if "attraction" in pl or ("restaurant" in pl and "budget" not in pl) or "nearby" in pl:
            return self._attractions_response(prompt)
        return self._itinerary_response(prompt)

    def _extract(self, prompt: str, field: str, default: str = "") -> str:
        """Extract a field value from prompt text."""
        patterns = {
            "destination": [r"Destination[:\s]+(\w+)", r"to\s+(\w+)"],
            "budget": [r"Budget[:\s]*[₹$]?(\d[\d,]*)", r"₹?(\d[\d,]*)"],
            "days": [r"(\d+)\s*(days?|nights?)", r"Days?[:\s]+(\d+)"],
            "interests": [r"Interests?[:\s]+(.+?)(?:\n|$)", r"(.+?)(?:adventure|nature)"],
        }
        if field in patterns:
            for pat in patterns[field]:
                m = re.search(pat, prompt, re.IGNORECASE)
                if m:
                    return m.group(1).strip()
        return default

    def _get_destination(self, prompt: str) -> str:
        dest = self._extract(prompt, "destination", "")
        if dest:
            return dest
        # Look for "for X" or "to X" pattern (common in template outputs)
        m = re.search(r'(?:for|to|in)\s+([A-Z][a-z]+)', prompt)
        if m:
            candidate = m.group(1)
            if candidate not in ("Budget", "Days", "Interests", "Travel", "Tips", "Packing", "Weather", "Temperature", "Based", "Please", "With", "Your"):
                return candidate
        # Fallback: look for capitalized place names
        places = re.findall(r'\b([A-Z][a-z]+)\b', prompt)
        exclude = {"Budget", "Days", "Destination", "Interests", "Travel", "Tips", "Packing", "Weather", "Temperature", "Based", "Visit", "Please", "Generate", "Create", "Provide", "With", "Your", "This", "That", "More", "From", "Local", "Morning", "Evening", "Early", "Afternoon", "Night", "Day", "Arrive", "Shopping", "Market", "Lunch", "Dinner"}
        for p in places:
            if p not in exclude:
                return p
        return "Manali"

    def _weather_response(self, prompt: str) -> str:
        city = self._get_destination(prompt)
        data = {
            "manali": ("18°C", "Partly Cloudy", "65%", "8 km/h", "16-22°C", "Light rain", "13-19°C"),
            "goa": ("32°C", "Sunny", "75%", "12 km/h", "30-34°C", "Clear", "28-33°C"),
            "shimla": ("16°C", "Clear Sky", "55%", "5 km/h", "14-20°C", "Cloudy", "12-18°C"),
            "delhi": ("35°C", "Haze", "45%", "10 km/h", "33-38°C", "Clear", "32-37°C"),
            "jaipur": ("34°C", "Sunny", "30%", "7 km/h", "32-38°C", "Clear", "30-36°C"),
            "kerala": ("29°C", "Humid", "80%", "6 km/h", "27-31°C", "Rain", "26-30°C"),
        }
        key = city.strip().lower()
        info = data.get(key, ("25°C", "Fair", "50%", "10 km/h", "22-28°C", "Fair", "20-27°C"))
        return f"""🌤️ Current Weather for {city}

Temperature   : {info[0]}
Condition     : {info[1]}
Humidity      : {info[2]}
Wind          : {info[3]}

📅 3-Day Forecast:
• Today:     {info[4]}, {info[1]}
• Tomorrow:  {info[5]}
• Day 3:     {info[6]}

💡 Tip: Check local weather apps for real-time updates during your trip."""

    def _currency_response(self, prompt: str) -> str:
        # Extract amount
        nums = re.findall(r'[\d,]+', prompt)
        amt = float(nums[0].replace(",", "")) if nums else 20000
        rates = {"USD": 83.15, "EUR": 89.75, "GBP": 104.20, "JPY": 0.56, "AUD": 54.80}
        lines = [f"🇮🇳 INR : ₹{amt:,.2f}"]
        for code, rate in rates.items():
            lines.append(f"🇺🇳 {code} : {amt/rate:,.2f}")
        lines.append(f"\n💡 Rate: 1 USD = ₹83.15 (approximate)")
        return "💱 Currency Conversion\n" + "\n".join(lines)

    def _attractions_response(self, prompt: str) -> str:
        city = self._get_destination(prompt)
        places = {
            "manali": (
                "🏔️ Solang Valley (14 km) - Skiing, Paragliding\n"
                "🏛️ Hadimba Temple (2 km) - Ancient Wood Temple\n"
                "🌊 Jogini Waterfall (3 km) - Scenic Trek\n"
                "🌲 Van Vihar Park (1 km) - Nature Walks",
                "1. The Lazy Dog - Continental & Indian (₹500-800)\n"
                "2. Cafe 1947 - Riverside Dining (₹400-700)\n"
                "3. Johnson's Cafe - Tibetan & Italian (₹300-600)"
            ),
            "goa": (
                "🏖️ Baga Beach - Water Sports\n"
                "🏛️ Basilica of Bom Jesus - Heritage\n"
                "🌴 Dudhsagar Falls - Trek & Swim\n"
                "🛕 Shree Mangeshi Temple - Culture",
                "1. Fisherman's Wharf - Seafood (₹800-1500)\n"
                "2. Thalassa - Greek (₹1000-2000)\n"
                "3. Gunpowder - South Indian (₹400-800)"
            ),
        }
        key = city.strip().lower()
        if key in places:
            attr, rest = places[key]
            return f"📍 Attractions in {city}\n\n{attr}\n\n🍽️ Restaurants\n{rest}"
        return f"📍 Popular attractions in {city}\nVisit local tourism websites for details."

    def _budget_response(self, prompt: str) -> str:
        dest = self._get_destination(prompt)
        # Extract budget after "Budget:" or "₹" specifically
        budget_m = re.search(r'[Bb]udget[:\s]*[₹$]?([\d,]+)', prompt)
        budget = int(budget_m.group(1).replace(",", "")) if budget_m else 20000
        days_m = re.search(r'(\d+)\s*(days?|nights?)', prompt)
        days = int(days_m.group(1)) if days_m else 4

        categories = [
            ("🏨 Accommodation", 0.20),
            ("🚌 Transport", 0.18),
            ("🍜 Food & Drinks", 0.15),
            ("🎟️ Activities", 0.22),
            ("🛍️ Shopping", 0.10),
            ("📋 Entry Fees", 0.05),
            ("💰 Contingency", 0.10),
        ]
        lines = [f"💰 Budget Breakdown for {dest} (₹{budget:,})"]
        lines.append(f"{'='*40}")
        for name, pct in categories:
            amt = budget * pct
            lines.append(f"{name:<20} ₹{amt:>8,.0f}  {pct*100:>3.0f}%")
        lines.append(f"{'─'*40}")
        lines.append(f"{'Total Budget':<20} ₹{budget:>8,}  100%")
        lines.append(f"\n📌 Per Day: ~₹{budget//days:,}")
        lines.append(f"💡 Save by booking early & using local transport!")
        return "\n".join(lines)

    def _tips_response(self, prompt: str) -> str:
        dest = self._get_destination(prompt)
        return f"""🧳 Packing Suggestions for {dest}

👕 Warm layers (thermals, fleece, jacket)
🥾 Comfortable walking/trekking shoes
🧴 Sunscreen SPF 50+ & sunglasses
💧 Reusable water bottle
📱 Power bank & chargers
💊 Basic first-aid kit
🆔 ID proof & travel documents

🚌 Local Travel Tips
• Use public transport to save money
• Eat at local dhabas for authentic food
• Carry cash for small vendors
• Negotiate prices at local markets
• Download offline maps
• Keep emergency numbers handy
• Try the local cuisine
• Respect local customs & dress codes

⚡ Pro Tip: Start your days early to avoid crowds!"""

    def _itinerary_response(self, prompt: str) -> str:
        dest = self._get_destination(prompt)
        nums = re.findall(r'\d+', prompt)
        days = int(nums[0]) if nums else 4
        days = min(max(days, 1), 14)

        interests_m = re.search(r'([Ii]nterests?[:\s]+)(.+?)(?:\n|$)', prompt)
        interests = interests_m.group(2).strip() if interests_m else "Adventure, Nature"

        itineraries = {
            1: [
                ("Day 1: Arrival & Exploration", [
                    "Arrive and check into hotel",
                    "Explore local market & Mall Road",
                    "Welcome dinner at local restaurant",
                ]),
            ],
            2: [
                ("Day 1: Arrival & Settle In", [
                    "Arrive at destination & check into hotel/homestay",
                    "Light exploration of nearby area",
                    "Visit local market for essentials",
                    "Welcome dinner (try local cuisine)",
                ]),
                ("Day 2: Adventure & Exploration", [
                    "Morning trek to scenic viewpoint/waterfall",
                    "Visit major attractions (temple/monastery/park)",
                    "Packed lunch amidst nature",
                    "Evening cultural show or local market walk",
                    "Bonfire dinner (if available)",
                ]),
            ],
            3: [
                ("Day 1: Arrival", [
                    "Arrive & check into hotel/homestay",
                    "Evening walk on Mall Road / local market",
                    "Welcome dinner at rooftop café",
                ]),
                ("Day 2: Main Sightseeing", [
                    "Full-day trip to major attractions",
                    "Visit viewpoints & photo stops",
                    "Lunch at local dhaba",
                    "Evening leisure time",
                ]),
                ("Day 3: Adventure & Departure", [
                    "Morning adventure activity (trekking / river crossing)",
                    "Last-minute souvenir shopping",
                    "Check out & depart",
                ]),
            ],
            4: [
                ("Day 1: Arrival & Local Exploration", [
                    "Arrive & check into hotel/homestay",
                    "Visit local market & nearby attractions",
                    "Evening walk on Mall Road",
                    "Dinner at a local café",
                ]),
                ("Day 2: Adventure Day", [
                    "Early morning trek to waterfall (3-4 hrs)",
                    "River crossing & rock climbing",
                    "Packed lunch by riverside",
                    "Visit temple/monastery",
                    "Bonfire at campsite",
                ]),
                ("Day 3: Nature & Sightseeing", [
                    "Full-day trip to Solang Valley / scenic spot",
                    "Adventure sports (skiing, paragliding)",
                    "Photography session at viewpoints",
                    "Local Himachali dinner",
                ]),
                ("Day 4: Departure", [
                    "Sunrise visit to hilltop viewpoint",
                    "Shopping for souvenirs",
                    "Check out & return journey",
                ]),
            ],
        }

        if days > 4:
            base = itineraries.get(4, itineraries[4])
            # Extend to more days with variations
            extra_days = []
            for d in range(5, days + 1):
                if d % 2 == 0:
                    extra_days.append((f"Day {d}: Nature & Relaxation", [
                        "Leisurely morning at a café",
                        "Nature walk / bird watching",
                        "Spa or wellness session",
                        "Explore offbeat trails",
                        "Evening stroll & local food",
                    ]))
                else:
                    extra_days.append((f"Day {d}: Adventure & Culture", [
                        "Visit local village / heritage site",
                        "Adventure activity (zip-line, rafting)",
                        "Cooking class / cultural workshop",
                        "Sunset photography session",
                    ]))
            base = base + extra_days
            itineraries[days] = base
        else:
            base = itineraries.get(days, itineraries[4])

        output = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🌍 {days}-DAY {dest.upper()} ITINERARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📅 Duration  : {days} Days / {days-1} Nights
🎯 Interests : {interests}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        for day_title, activities in base:
            output += f"\n{day_title}\n"
            output += "─" * 40 + "\n"
            for act in activities:
                output += f"  ✅ {act}\n"

        output += """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Your personalized itinerary is ready!
💡 Use options menu to modify, export, or get more info.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return output

    @property
    def _llm_type(self) -> str:
        return "mock-travel-planner"


# =========================================================================
# TOOLS (LangChain @tool decorator)
# =========================================================================

@tool
def get_weather(city: str) -> str:
    """Get current weather information for any city. Useful when users ask about weather conditions, temperature, or forecast."""
    mock = MockLLM()
    return mock._weather_response(f"Weather in {city}")


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another using current exchange rates."""
    pl = f"Convert {amount} {from_currency} to {to_currency}"
    mock = MockLLM()
    return mock._currency_response(pl)


@tool
def get_attractions(city: str) -> str:
    """Get nearby attractions, tourist spots, and recommended restaurants for any city."""
    mock = MockLLM()
    return mock._attractions_response(f"Attractions in {city}")


# =========================================================================
# PROMPT TEMPLATES
# =========================================================================

ITINERARY_TEMPLATE = PromptTemplate(
    input_variables=["destination", "days", "budget", "interests", "dates"],
    template="itinerary for destination: {destination}, days: {days}, interests: {interests}"
)

BUDGET_TEMPLATE = PromptTemplate(
    input_variables=["destination", "days", "budget", "itinerary"],
    template="budget breakdown for destination: {destination}, days: {days}, budget: {budget}"
)

TIPS_TEMPLATE = PromptTemplate(
    input_variables=["destination", "interests"],
    template="travel tips and packing for destination: {destination}, interests: {interests}"
)


# =========================================================================
# MAIN APPLICATION
# =========================================================================

@dataclass
class TravelData:
    destination: str = ""
    days: int = 1
    budget: str = ""
    interests: str = ""
    dates: str = ""


class TravelPlanner:
    """AI Travel Planner with LangChain components."""

    def __init__(self, mock_mode: bool = True):
        self.mock_mode = mock_mode
        self.mock_llm = MockLLM() if mock_mode else self._get_real_llm()
        # Wrap in RunnableLambda to satisfy LangChain 0.3+ Runnable interface
        self.llm = RunnableLambda(lambda msg: self.mock_llm.invoke(msg))
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.travel = TravelData()
        self.tools = [get_weather, convert_currency, get_attractions]

        # Build chains using LangChain's pipe operator (modern pattern)
        self.itinerary_chain = ITINERARY_TEMPLATE | self.llm
        self.budget_chain = BUDGET_TEMPLATE | self.llm
        self.tips_chain = TIPS_TEMPLATE | self.llm

    def _get_real_llm(self):
        """Initialize real LLM (Groq is free - no credit card needed)."""
        try:
            from langchain_groq import ChatGroq
            api_key = os.getenv("GROQ_API_KEY", "")
            if api_key:
                return ChatGroq(temperature=0.7, groq_api_key=api_key, model_name="mixtral-8x7b-32768")
        except ImportError:
            pass
        print(" Groq not configured. Using mock mode.")
        return MockLLM()

    def _call_tools(self, query: str) -> str:
        """Demonstrate automatic tool calling based on query content."""
        results = []
        city = self.travel.destination or "Manali"

        if any(w in query.lower() for w in ["weather", "temperature", "climate", "rain", "forecast", "humidity"]):
            results.append(f"[Tool Called: get_weather]\n{get_weather.invoke({'city': city})}")

        if any(w in query.lower() for w in ["currency", "convert", "exchange", "rate", "dollar", "euro", "money"]):
            budget = self.travel.budget.replace(",", "")
            try:
                amt = float(budget) if budget else 20000
            except ValueError:
                amt = 20000
            results.append(f"[Tool Called: convert_currency]\n{convert_currency.invoke({'amount': amt, 'from_currency': 'INR', 'to_currency': 'USD'})}")

        if any(w in query.lower() for w in ["attraction", "restaurant", "nearby", "place", "visit", "food", "eat", "sightsee"]):
            results.append(f"[Tool Called: get_attractions]\n{get_attractions.invoke({'city': city})}")

        return "\n\n".join(results)

    def collect_details(self) -> TravelData:
        """Collect travel details via CLI."""
        print("\n" + "=" * 60)
        print("   AI TRAVEL PLANNER   ".center(60))
        print("=" * 60)
        print("\nEnter your travel details:\n")

        self.travel.destination = input("  Destination  : ").strip() or "Manali"
        days_input = input("  Days         : ").strip()
        self.travel.days = int(days_input) if days_input.isdigit() else 4
        self.travel.budget = input("  Budget (INR) : ").strip() or "20000"
        self.travel.interests = input("  Interests    : ").strip() or "Adventure, Nature"
        self.travel.dates = input("  Dates        : ").strip() or "Not specified"

        return self.travel

    def generate_plan(self) -> Dict[str, str]:
        """Run chains sequentially to generate itinerary, budget, and tips."""
        print("\nGenerating your personalized travel plan...\n")

        # Tool calling demo
        query = f"Trip to {self.travel.destination}, weather, currency, attractions"
        print("Automatic Tool Calling Demo:")
        tools_output = self._call_tools(query)
        if tools_output:
            print(tools_output)
            self.memory.chat_memory.add_ai_message(tools_output)

        inputs = {
            "destination": self.travel.destination,
            "days": self.travel.days,
            "budget": self.travel.budget,
            "interests": self.travel.interests,
            "dates": self.travel.dates,
        }

        # Chain 1: Generate itinerary
        print("\nGenerating itinerary...")
        itinerary_result = self.itinerary_chain.invoke(inputs)
        itinerary = itinerary_result.content if hasattr(itinerary_result, 'content') else str(itinerary_result)
        self.memory.chat_memory.add_ai_message(f"Itinerary:\n{itinerary}")

        # Chain 2: Estimate budget (using itinerary from chain 1)
        print("Estimating budget...")
        budget_inputs = {**inputs, "itinerary": itinerary}
        budget_result = self.budget_chain.invoke(budget_inputs)
        budget_info = budget_result.content if hasattr(budget_result, 'content') else str(budget_result)
        self.memory.chat_memory.add_ai_message(f"Budget:\n{budget_info}")

        # Chain 3: Travel tips
        print("Getting travel tips...")
        tips_inputs = {"destination": self.travel.destination, "interests": self.travel.interests}
        tips_result = self.tips_chain.invoke(tips_inputs)
        tips = tips_result.content if hasattr(tips_result, 'content') else str(tips_result)
        self.memory.chat_memory.add_ai_message(f"Tips:\n{tips}")

        return {
            "itinerary": itinerary,
            "budget_info": budget_info,
            "tips": tips
        }

    def display_plan(self, output: Dict[str, str]):
        """Display the complete travel plan."""
        print("\n" + "=" * 60)
        print("YOUR PERSONALIZED TRAVEL PLAN".center(60))
        print("=" * 60)

        print("\n" + output.get("itinerary", ""))

        print("\n" + "=" * 60)
        print("BUDGET ESTIMATE".center(60))
        print(output.get("budget_info", ""))

        print("\n" + "=" * 60)
        print("TRAVEL TIPS".center(60))
        print(output.get("tips", ""))

    def modify_trip(self, modification: str) -> bool:
        """Handle trip modifications using memory context."""
        print(f"\n🔄 Processing: {modification}")
        mod_lower = modification.lower()
        changed = False

        # Parse budget change
        nums = re.findall(r'\d[\d,]*', modification)
        if "budget" in mod_lower and nums:
            self.travel.budget = nums[-1].replace(",", "")
            print(f"✅ Budget updated to ₹{self.travel.budget}")
            changed = True

        if "day" in mod_lower or "night" in mod_lower:
            if nums:
                self.travel.days = int(nums[0])
                print(f"✅ Duration updated to {self.travel.days} days")
                changed = True

        # Parse destination change
        for word in modification.split():
            w = word.strip(".,!?")
            if w.lower() in ("to", "destination", "dest") and w != modification.split()[-1]:
                idx = modification.split().index(word)
                if idx + 1 < len(modification.split()):
                    candidate = modification.split()[idx + 1].strip(".,!?")
                    if candidate[0].isupper() and len(candidate) > 2:
                        self.travel.destination = candidate
                        print(f"✅ Destination updated to {candidate}")
                        changed = True
                        break

        if not changed:
            # Try to find a capitalized word that could be a destination
            for word in modification.split():
                w = word.strip(".,!?")
                if w.istitle() and len(w) > 2 and w.lower() not in ("budget", "days", "destination", "interests", "to", "the", "my", "add", "change", "update", "modify"):
                    self.travel.destination = w
                    print(f"✅ Destination updated to {w}")
                    changed = True
                    break

        if not changed:
            print("❌ Could not understand the modification. Try: 'Change budget to 25000' or 'Add 2 more days'")
            return False

        # Regenerate with updated data
        print("🔄 Regenerating plan with updated details...")
        output = self.generate_plan()
        self.display_plan(output)
        return True

    def export_plan(self, filename: str = "my_travel_plan.txt"):
        """Export the full plan to a text file."""
        os.makedirs("exports", exist_ok=True)
        path = os.path.join("exports", filename)

        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("AI TRAVEL PLANNER - ITINERARY\n".center(60))
            f.write("=" * 60 + "\n\n")
            f.write(f"Destination : {self.travel.destination}\n")
            f.write(f"Days        : {self.travel.days}\n")
            f.write(f"Budget      : ₹{self.travel.budget}\n")
            f.write(f"Interests   : {self.travel.interests}\n")
            f.write(f"Dates       : {self.travel.dates}\n\n")

            # Write memory contents
            for msg in self.memory.chat_memory.messages:
                role = "USER" if isinstance(msg, HumanMessage) else "AI"
                f.write(f"\n[{role}]\n{msg.content}\n")

        print(f"\n✅ Exported to {os.path.abspath(path)}")

    def run(self):
        """Main application loop."""
        self.collect_details()

        output = self.generate_plan()
        self.display_plan(output)

        # Interactive options menu
        while True:
            print("\n" + "=" * 60)
            print("OPTIONS MENU".center(60))
            print("=" * 60)
            print("  1. 🔄 Modify trip (budget, days, destination)")
            print("  2. 📄 Export itinerary to text file")
            print("  3. 💱 Convert currency")
            print("  4. 🌤️  Check weather")
            print("  5. 📍 Find nearby attractions & restaurants")
            print("  6. ❌  Exit")

            choice = input("\n🔢 Your choice (1-6): ").strip()

            if choice == "1":
                mod = input("✏️  What to change? (e.g., 'Budget 25000', 'Add 2 days', 'Change to Goa'): ").strip()
                if mod:
                    self.modify_trip(mod)
                else:
                    print("❌ No modification entered.")
            elif choice == "2":
                fname = input("📁 Filename (default: my_travel_plan.txt): ").strip()
                self.export_plan(fname or "my_travel_plan.txt")
            elif choice == "3":
                amt = input("💰 Amount: ").strip() or "20000"
                fr = input("🔤 From (e.g., INR, USD): ").strip().upper() or "INR"
                to = input("🔤 To: ").strip().upper() or "USD"
                print("\n" + convert_currency.invoke({"amount": float(amt.replace(",", "")), "from_currency": fr, "to_currency": to}))
            elif choice == "4":
                city = input("📍 City: ").strip() or self.travel.destination
                print("\n" + get_weather.invoke({"city": city}))
            elif choice == "5":
                city = input("📍 City: ").strip() or self.travel.destination
                print("\n" + get_attractions.invoke({"city": city}))
            elif choice == "6":
                print("\n👋 Thank you for using AI Travel Planner! Safe travels! 🎉")
                break
            else:
                print("❌ Invalid choice. Please enter 1-6.")


# =========================================================================
# ENTRY POINT
# =========================================================================

def main():
    print("=" * 60)
    print("🌍  WELCOME TO AI TRAVEL PLANNER  🌍".center(60))
    print("=" * 60)
    print("\n⚙️  Mode: MOCK (no API keys required, works offline)")
    print("💡 For real AI: Get free key at console.groq.com")
    print("   Then: set GROQ_API_KEY=your_key & change mock_mode=False")

    planner = TravelPlanner(mock_mode=True)
    try:
        planner.run()
    except KeyboardInterrupt:
        print("\n\n👋 Thanks for using AI Travel Planner! Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
