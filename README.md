Sustainable Vendor Decision System (V4_VDS.py)
Created by: CogitoCore

🏭 Multi-Agent AI Platform for Enterprise Procurement Optimization

## 🌐 Live Demo

**Try it now**: [https://your-app-url.streamlit.app](https://your-app-url.streamlit.app)

> 🎭 The live demo runs in **Demo Mode** by default (no API keys required)  
> 🔐 Login credentials: Username: `CogitoCore` | Password: [any value]

**Quick Start**: Click the link → Login → Navigate to "New Evaluation" → Select vendors → See AI-powered results!

---

📋 Table of Contents
Overview
Key Features
Architecture
Installation
Configuration
Usage Guide
Multi-Agent System
API Integration
Demo Mode
System Requirements
Troubleshooting
Project Structure
Contributing
Acknowledgments
🎯 Overview

Sustainable Vendor Decision System is an advanced Multi-Agent AI System designed for enterprise purchasing teams to optimize vendor selection with a focus on supply chain sustainability. Built specifically for the textile industry purchasing department, the system leverages Google Gemini LLM, TOPSIS multi-criteria decision analysis, and intelligent agent orchestration to evaluate and rank vendors across multiple dimensions.

Business Context
Target Users: Purchasing departments, procurement managers, supply chain analysts
Industry: Textile manufacturing and retail
Objective: Optimize vendor selection while ensuring ESG compliance and supply chain resilience
What Makes It Unique?

✅ LLM-Powered Intelligence - Uses Google Gemini 1.5 Flash for contextual sustainability analysis
✅ Multi-Agent Architecture - Six specialized agents working in parallel and sequential workflows
✅ Validation Loops - Automatic iterative refinement to meet sustainability thresholds
✅ Real-Time Data Enrichment - Integrates Google Custom Search for live vendor intelligence
✅ Full Observability - Comprehensive logging, tracing, and metrics tracking
✅ Interactive UI - Streamlit-based dashboard for non-technical users
✅ Demo Mode - Works without API keys for testing and education

🚀 Key Features
🤖 Multi-Agent Intelligence
Agent	Role	LLM-Powered
DataCollectionAgent	Web search & data enrichment	❌
SustainabilityAgent	ESG scoring (carbon, labor, waste)	✅
RiskAnalysisAgent	Supply chain risk assessment	✅
TOPSISRankingAgent	Multi-criteria decision analysis	❌
ValidationAgent	Quality assurance with feedback loops	❌
MemoryAgent	Long-term storage & context retrieval	✅
📊 Decision Criteria

The system evaluates vendors across 5 key dimensions:

💰 Cost - Total procurement cost
⭐ Quality - Product/service quality score (0-100)
🚚 Delivery Time - Lead time in days
⚠️ Risk - Supply chain risk score (0-100)
🌱 Sustainability - Composite ESG score
Carbon footprint management
Labor practices & ethics
Waste & resource management
🔧 Tools & Integrations
Google Custom Search API - Real-time vendor intelligence gathering
Google Gemini 1.5 Flash - Natural language reasoning for sustainability
Google Sheets API - Persistent evaluation storage (optional)
TOPSIS Algorithm - Mathematical multi-criteria optimization
ThreadPoolExecutor - Parallel agent execution
📈 Observability
JSON Structured Logging - File + console output (vds_system.log)
Distributed Tracing - Message bus tracks all A2A communications
Performance Metrics - Agent-level execution times and success rates
Validation Audit Trail - Complete record of iterative weight adjustments
🏗️ Architecture
System Diagram
┌─────────────────────────────────────────┐
│         Streamlit UI Layer              │
│   (Session Management & Visualization)  │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│      Multi-Agent Orchestrator           │
│  (Lifecycle Management & Coordination)  │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│          Message Bus (A2A)              │
│   (Publish/Subscribe Communication)     │
└───┬─────────┬─────────┬─────────┬───────┘
    │         │         │         │
┌───▼───┐ ┌──▼───┐ ┌───▼──┐ ┌────▼────┐
│ Data  │ │ Risk │ │ Sust.│ │ TOPSIS  │
│Collect│ │Analys│ │Agent │ │ Ranking │
└───┬───┘ └──┬───┘ └───┬──┘ └────┬────┘
    │        │         │         │
    └────────┴─────────┴─────────┘
              │
         ┌────▼─────┐
         │Validation│ ← [Feedback Loop]
         │  Agent   │
         └────┬─────┘
              │
         ┌────▼─────┐
         │  Memory  │
         │  Agent   │
         └──────────┘

Execution Flow
Phase 1 (Parallel): DataCollection + RiskAnalysis execute concurrently
Phase 2 (Sequential): SustainabilityAgent performs LLM-based ESG scoring
Phase 3 (Sequential): TOPSISAgent calculates weighted rankings
Phase 4 (Loop): ValidationAgent iterates until sustainability threshold met (max 3 loops)
Phase 5 (Persistence): MemoryAgent saves evaluation to long-term storage
📦 Installation
Prerequisites
Python 3.8+
pip package manager
Internet connection (for API calls)
Quick Install
# Clone or download Sustainable Vendor Decision System.py
cd /path/to/project

# Install dependencies
pip install google-generativeai google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client pandas numpy streamlit plotly requests

# Or use requirements.txt if provided
pip install -r requirements.txt

Create requirements.txt (Optional)
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.14.0
google-generativeai>=0.3.0
google-auth>=2.16.0
google-auth-oauthlib>=1.0.0
google-auth-httplib2>=0.1.0
google-api-python-client>=2.80.0
requests>=2.28.0

⚙️ Configuration
API Keys Setup

The system supports three API integrations (all optional with demo mode):

1. 🤖 Google Gemini API

Purpose: LLM reasoning for sustainability and risk analysis

How to obtain:

Visit Google AI Studio
Sign in with Google account
Click "Create API Key"
Copy the key (format: AIzaSy...)

Cost: Free tier available (60 requests/minute)

2. 🔍 Google Custom Search API

Purpose: Real-time vendor intelligence gathering

How to obtain:

Step A - API Key:

Go to Google Cloud Console
Create a new project or select existing
Click "Create Credentials" → "API key"
Enable "Custom Search API" in API Library
Copy the API key

Step B - Search Engine ID:

Visit Programmable Search Engine
Click "Add" to create new search engine
Set "Search the entire web" option
Copy the "Search Engine ID" (format: 017576662...)

Cost: 100 free queries/day, then $5 per 1000 queries

3. 📊 Google Sheets API (Optional)

Purpose: Long-term evaluation storage

Setup: Requires service account credentials (JSON file) - advanced configuration not covered in basic setup.

🎮 Usage Guide
Starting the Application
# Navigate to project directory
cd /path/to/your/project

# Run Streamlit app
streamlit run Sustainable Vendor Decision System.py


The app will open in your browser at http://localhost:8501

Step-by-Step Workflow
1. Login Page

Credentials:

Username: CogitoCore
Password: [Any value - not validated in demo]

API Configuration:

Enter your API keys (Gemini, Search API, Search Engine ID)
OR leave blank for Demo Mode

Demo Mode Indicator:

🎭 Demo Mode Active - Uses simulated data
✅ Live Mode - Real API integration enabled

Click 🚀 Sign In & Initialize System

2. Dashboard

Overview Metrics:

Total Vendors
Evaluations Run
Active Agents
Messages Exchanged

Recent Evaluations: Browse past evaluation results

Navigation: Use sidebar to access different pages

3. Vendor Management

Pre-loaded Vendors (6 default):

Global Textiles Ltd
EcoFabrics Inc
Premium Weave Co
Sustainable Threads
FastFabric Solutions
Quality First Textiles

Actions:

View vendor details in table format
Add new vendors with custom data
Delete vendors by ID
4. New Evaluation (Main Workflow)

Step 1 - Select Vendors:

Check vendors to include in evaluation
Minimum 1 vendor required
Click Next →

Step 2 - Configure Weights:

Choose preset: Balanced, Cost Focused, Sustainability First, Quality First
Or adjust custom weights using sliders:
💰 Cost (default: 0.2)
⭐ Quality (default: 0.2)
🚚 Delivery (default: 0.2)
⚠️ Risk (default: 0.2)
🌱 Sustainability (default: 0.2)
Weights are auto-normalized to sum = 1.0
Click Next →

Step 3 - Execute:

Review selection and weights
Enter evaluation name (auto-generated by default)
Enable/disable parallel execution
Click 🚀 Execute

Processing (15-30 seconds):

Phase 1: Data collection & enrichment...
Phase 2: Sustainability analysis...
Phase 3: TOPSIS ranking & validation...
✅ Complete!
5. Results Page

Summary Metrics:

Total execution time
Messages exchanged
Validation loops executed
Number of vendors evaluated

Rankings Table:

TOPSIS Score (0-1, higher is better)
Sustainability Score (0-100)
All criteria values

🥇 Recommended Vendor: Top-ranked vendor highlighted

Tabs:

📊 Performance Radar: Visual comparison of top 3 vendors
🌱 Sustainability: Detailed ESG breakdown with Gemini reasoning
📈 Agent Metrics: Execution time per agent
🔍 Validation Log: Audit trail of weight adjustments

Export: Download results as CSV

6. System Metrics

Agent Status Table:

Total runs per agent
Average execution duration
Success rate (%)

Message Bus Activity:

Total messages exchanged
Recent message log (last 20)

Performance Trends:

Line chart of evaluation durations over time
7. API Settings

View Current Configuration:

Gemini API status (✅ Configured / 🎭 Demo)
Search API status

Update Credentials:

Enter new API keys
Click 💾 Save & Reinitialize System
System will reload with new configuration

Quick Links: Direct access to API provider portals

🤖 Multi-Agent System
Agent Details
DataCollectionAgent
Type: Parallel execution
Function: Web search for vendor certifications and compliance evidence
Tool: GoogleSearchTool
Output: vendor.evidence_found, vendor.certifications
Message: Publishes data_enrichment to SustainabilityAgent
SustainabilityAgent ⭐ LLM-Powered
Type: Sequential execution
Function: AI-driven ESG analysis
Tool: Google Gemini 1.5 Flash
Prompt Strategy:
Few-shot examples of certification scoring
JSON response format enforcement
Fallback heuristics if LLM unavailable
Output: carbon_score, labor_score, waste_score, sustainability_score
Message: Publishes sustainability_score to RiskAnalysisAgent
RiskAnalysisAgent ⭐ LLM-Powered
Type: Parallel execution
Function: Supply chain risk narrative generation
Tool: Google Gemini 1.5 Flash
Prompt Strategy: Multi-dimensional risk categorization (delivery, compliance, reputational, financial)
Output: vendor.risk_analysis
Message: Publishes risk_analysis to TOPSISAgent
TOPSISRankingAgent
Type: Sequential execution
Function: Mathematical multi-criteria optimization
Algorithm: TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
Process:
Normalize decision matrix
Apply weights to criteria
Calculate Euclidean distance to ideal best/worst solutions
Compute relative closeness score
Output: vendor.topsis_score, sorted vendor list
Message: Publishes ranking to ValidationAgent
ValidationAgent ⭐ Loop Agent
Type: Iterative loop (max 3 iterations)
Function: Quality assurance with automatic weight adjustment
Logic:
while iteration < 3:
    if top_vendor.sustainability_score >= 60.0:
        ✓ PASS
    else:
        weights['sustainability'] += 0.15
        re-run TOPSIS

Output: Final validated rankings, adjusted weights, audit log
MemoryAgent
Type: Utility / storage
Function: Long-term evaluation persistence and vendor history retrieval
Tool: GoogleSheetsTool (with in-memory fallback)
LLM Feature: Uses Gemini to summarize historical vendor performance
Output: Historical context for vendor decisions
Message Bus (A2A Protocol)

Structured Communication Format:

@dataclass
class AgentMessage:
    sender: str           # Agent name
    recipient: str        # Target agent
    payload: Dict         # Data payload
    timestamp: float      # Unix timestamp
    message_type: str     # data_enrichment | sustainability_score | 
                          # ranking | risk_analysis | validation


Benefits:

Decoupled agent communication
Complete audit trail
Supports future agent additions without code changes
🔌 API Integration
API Call Patterns
Google Gemini

Request Pattern:

prompt = f"""
You are a sustainability analyst.
Analyze vendor: {vendor.name}
Certifications: {vendor.certifications}
Evidence: {vendor.evidence_found}

Provide scores (0-100):
- carbon_score
- labor_score  
- waste_score

Return as JSON with reasoning.
"""

response = gemini_model.generate_content(prompt)
scores = json.loads(response.text)


Retry Logic: 3 attempts with 1-second backoff

Fallback: Heuristic scoring based on certification keywords

Google Custom Search

Request Pattern:

query = f"{vendor.name} textile sustainability certifications compliance"
service = build("customsearch", "v1", developerKey=api_key)
result = service.cse().list(q=query, cx=engine_id, num=5).execute()

results = [item['title'], item['snippet'], item['link'] 
           for item in result['items']]


Fallback: Pre-loaded knowledge base with simulated search results for default vendors

Rate Limits & Quotas
API	Free Tier	Paid Tier
Gemini 1.5 Flash	60 req/min	1000 req/min
Custom Search	100 req/day	10,000 req/day
Sheets API	Unlimited (with quotas)	Unlimited

Cost Estimation (per evaluation):

Gemini calls: 2-3 per vendor (sustainability + risk)
Search calls: 1 per vendor
Example: 6 vendors = ~15 Gemini calls + 6 search calls
Daily capacity (free tier): ~6 evaluations/day
🎭 Demo Mode
What is Demo Mode?

When API keys are not provided or set to "demo_mode", the system operates with:

✅ Simulated Search Results - Pre-loaded knowledge base for 6 default vendors
✅ Heuristic ESG Scoring - Rule-based sustainability calculation using certification keywords
✅ In-Memory Storage - Evaluations stored in session (not persisted)
✅ Full Feature Access - All UI features and workflows available

When to Use Demo Mode?
✅ Educational purposes / learning the system
✅ Testing workflows without API costs
✅ Presenting to stakeholders (no internet required)
✅ Development and debugging
Limitations
❌ No real-time vendor intelligence
❌ Less sophisticated ESG reasoning (no LLM nuance)
❌ No persistence across sessions
❌ Fixed knowledge base (6 vendors only)
Enabling Demo Mode

Option 1: Leave all API fields empty during login

Option 2: Enter demo_mode in any API key field

Indicator: Look for 🎭 Demo Mode Active badge in UI

💻 System Requirements
Hardware
CPU: 2+ cores recommended (for parallel agent execution)
RAM: 4 GB minimum, 8 GB recommended
Storage: 500 MB (includes dependencies)
Network: Broadband internet (for API calls)
Software
Operating System: Windows 10+, macOS 10.14+, Linux (Ubuntu 20.04+)
Python: 3.8, 3.9, 3.10, 3.11 (tested)
Browser: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
Python Dependencies
streamlit (1.28+)      - Web UI framework
pandas (1.5+)          - Data manipulation
numpy (1.24+)          - Numerical computing
plotly (5.14+)         - Interactive visualizations
google-generativeai    - Gemini API client
google-api-python-client - Google services

🔧 Troubleshooting
Common Issues
1. Import Errors

Symptom: ModuleNotFoundError: No module named 'streamlit'

Solution:

pip install --upgrade streamlit pandas numpy plotly google-generativeai google-api-python-client

2. Gemini API Errors

Symptom: [Gemini API unavailable - using fallback analysis] in results

Possible Causes:

Invalid API key
Rate limit exceeded (60 req/min)
Network connectivity issues

Solution:

Verify API key at Google AI Studio
Check rate limits in Google Cloud Console
Wait 60 seconds if rate limited
System automatically falls back to heuristic scoring
3. Search API Quota Exceeded

Symptom: Demo mode search results even with valid API key

Solution:

Check daily quota (100 free queries/day)
Wait until midnight UTC for quota reset
Consider upgrading to paid tier
Use demo mode temporarily
4. Streamlit Won't Start

Symptom: Address already in use error

Solution:

# Kill existing Streamlit process
pkill -f streamlit

# Or use different port
streamlit run Sustainable Vendor Decision System.py --server.port 8502

5. Slow Performance

Symptom: Evaluation takes >60 seconds

Possible Causes:

Network latency to Google APIs
Sequential agent execution (disable parallel mode)
Large number of vendors (>10)

Solution:

Enable parallel execution in Step 3
Reduce number of vendors
Use demo mode for faster testing
Check internet connection speed
6. Validation Loop Stuck

Symptom: Validation never passes, 3 iterations always reached

Cause: No vendor meets sustainability threshold even after weight adjustment

Solution:

Lower threshold in code: ValidationAgent(bus, gemini_api_key, min_sustainability_threshold=40.0)
Add vendors with better ESG profiles
Check if Gemini analysis is working (not falling back to heuristics)
Debug Mode

Enable detailed logging:

# Add to top of Sustainable Vendor Decision System.py after imports
import logging
logging.basicConfig(level=logging.DEBUG)


Check log file:

cat vds_system.log

📁 Project Structure
project/
│
├── Sustainable Vendor Decision System.py                  # Main application file (1200+ lines)
│
├── vds_system.log              # Execution logs (auto-generated)
│
├── requirements.txt            # Python dependencies
│
├── README.md                   # This file
│
└── [Optional]
    ├── .env                    # API keys (not committed to git)
    ├── credentials.json        # Google Sheets service account
    └── evaluations/            # Exported CSV results

Code Organization (Sustainable Vendor Decision System.py)
Lines 1-20      │ Dependencies & imports
Lines 24-31     │ Logging configuration
Lines 35-48     │ Domain models (Vendor, AgentMessage, etc.)
Lines 57-73     │ Message Bus (A2A protocol)
Lines 76-130    │ BaseAgent class
Lines 193-290   │ Tools (GoogleSearch, GoogleSheets)
Lines 293-575   │ Agent implementations (6 agents)
Lines 577-609   │ MultiAgentOrchestrator
Lines 682-1100  │ Streamlit UI (8 pages)
Lines 1113-1135 │ Main entry point

🤝 Contributing
Development Setup
# Clone repository
git clone <repo-url>
cd svds-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dev dependencies
pip install -r requirements.txt

# Run tests (if implemented)
pytest tests/

Adding New Agents
class MyCustomAgent(BaseAgent):
    def __init__(self, bus: MessageBus, gemini_api_key: str = None):
        super().__init__("MyCustomAgent", bus, gemini_api_key)
    
    def run(self, vendors: List[Vendor]) -> List[Vendor]:
        @self.track_execution
        def _execute():
            self.log("Starting custom analysis...")
            
            # Your logic here
            for vendor in vendors:
                # Process vendor
                pass
            
            # Publish message
            message = AgentMessage(
                sender=self.name,
                recipient="NextAgent",
                payload={"data": "value"},
                timestamp=time.time(),
                message_type="custom_type"
            )
            self.bus.publish(message)
            
            return vendors
        
        return _execute()


Register in orchestrator:

self.custom_agent = MyCustomAgent(self.bus, gemini_api_key)
# Add to execution pipeline in run_evaluation()



🙏 Acknowledgments
Technologies
Google Gemini - Advanced LLM reasoning capabilities
Streamlit - Rapid UI development framework
Plotly - Interactive data visualizations
NumPy - High-performance numerical computing
Methodologies
TOPSIS - Multi-criteria decision making algorithm (Hwang & Yoon, 1981)
Multi-Agent Systems - Distributed AI architectures
ESG Framework - Environmental, Social, Governance criteria
Inspiration

This project was developed for enterprise procurement optimization, inspired by real-world supply chain sustainability challenges in the textile industry.

📞 Support & Contact
Documentation
API Documentation: Google AI Documentation
Streamlit Docs: docs.streamlit.io
Getting Help
Check Troubleshooting Section: See Troubleshooting
Review Logs: Check vds_system.log for detailed error messages
Demo Mode Testing: Try demo mode to isolate API vs. logic issues
Reporting Issues

When reporting bugs, include:

Python version (python --version)
Operating system
Full error traceback
API mode (live vs. demo)
Steps to reproduce
🎓 Educational Use

This system is designed for:

Agents Intensive - Capstone Project
Industry Training - Procurement optimization, ESG analysis
Proof of Concept - Demonstrating LLM integration in business processes


🗺️ Roadmap
Planned Features (v5.0)
 Advanced Memory: Vector database for vendor embeddings
 Multi-LLM Support: Gemini, Claude, GPT-4, Llama integration
 Batch Processing: Evaluate 100+ vendors simultaneously
 RESTful API: Headless operation via HTTP endpoints
 Real-time Monitoring: Live dashboard for running evaluations
 Custom Agents: UI-based agent builder (no code required)
 Export Formats: PDF reports, PowerPoint summaries
 Multi-language: Spanish, French, German, Chinese support
Research Directions
Reinforcement Learning: Agents learn from user feedback
Explainable AI: Enhanced transparency in LLM reasoning
Federated Learning: Privacy-preserving multi-org collaboration
📊 Performance Benchmarks
Typical Execution Times (6 vendors)
Mode	Parallel Agents	Total Duration	API Calls
Demo	Enabled	3-5 seconds	0
Live	Enabled	15-25 seconds	~18
Live	Disabled	25-40 seconds	~18
Scalability
Vendors	Execution Time	Memory Usage
5-10	15-30s	~150 MB
11-25	30-60s	~200 MB
26-50	60-120s	~300 MB
50+	Not recommended	Memory issues possible

Note: Performance depends on network latency and API response times.

Built with ❤️ for sustainable purchasing

⭐ Star this project if you find it useful! ⭐
