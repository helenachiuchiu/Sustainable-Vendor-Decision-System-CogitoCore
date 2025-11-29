# ============================================================================
# Sustainable Vendor Decision System
# Multi-Agent ADK System with Google Gemini & Real Tools
# Agents Intensive - Capstone Project (Kaggle · Community Hackathon)
# ============================================================================
# Installation (Run in Colab or terminal):
# pip install google-generativeai google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client pandas numpy streamlit plotly requests

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
import plotly.express as px

# Google Gemini
import google.generativeai as genai

# Google APIs
from googleapiclient.discovery import build
from google.oauth2 import service_account

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "agent": "%(name)s", "message": "%(message)s"}',
    handlers=[
        logging.FileHandler("vds_system.log"),
        logging.StreamHandler()
    ]
)

# ============================================================================
# DOMAIN MODELS
# ============================================================================

@dataclass
class Vendor:
    id: str
    name: str
    cost: float
    financial_stability: float
    lead_time: int
    technology: float
    quality: float
    hygiene: float
    supply_chain_risk: float
    certifications: str = "None"
    
    # Computed by agents
    carbon_score: float = 0.0
    labor_score: float = 0.0
    waste_score: float = 0.0
    ESG_score: float = 0.0
    topsis_score: float = 0.0
    risk_analysis: str = ""
    audit_log: str = ""
    evidence_found: str = ""
    gemini_reasoning: str = ""

@dataclass
class AgentMessage:
    """Structured message for A2A communication"""
    sender: str
    recipient: str
    payload: Dict[str, Any]
    timestamp: float
    message_type: str  # data_enrichment, ESG_score, ranking, validation

@dataclass
class EvaluationRecord:
    id: str
    timestamp: float
    name: str
    vendors_involved: List[str]
    weights: Dict[str, float]
    results: List[Dict]
    execution_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionMetrics:
    agent_name: str
    start_time: float
    end_time: float
    duration: float
    status: str
    error: Optional[str] = None

# ============================================================================
# MESSAGE BUS (A2A Protocol)
# ============================================================================

class MessageBus:
    """Central message passing system for agent communication"""
    def __init__(self):
        self.messages: List[AgentMessage] = []
        self.logger = logging.getLogger("MessageBus")
    
    def publish(self, message: AgentMessage):
        self.messages.append(message)
        self.logger.info(f"Message published: {message.sender} -> {message.recipient} ({message.message_type})")
    
    def get_messages_for(self, recipient: str, message_type: Optional[str] = None) -> List[AgentMessage]:
        filtered = [m for m in self.messages if m.recipient == recipient]
        if message_type:
            filtered = [m for m in filtered if m.message_type == message_type]
        return filtered
    
    def clear(self):
        self.messages.clear()

# ============================================================================
# BASE AGENT CLASS
# ============================================================================

class BaseAgent:
    """Enhanced base agent with observability and LLM integration"""
    def __init__(self, name: str, bus: MessageBus, gemini_api_key: str = None):
        self.name = name
        self.bus = bus
        self.logger = logging.getLogger(name)
        self.metrics: List[ExecutionMetrics] = []
        self.gemini_api_key = gemini_api_key
        
        # Initialize Gemini model if API key provided
        self.gemini_model = None
        if gemini_api_key and "demo_mode" not in gemini_api_key.lower():
            try:
                genai.configure(api_key=gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
            except Exception as e:
                self.logger.warning(f"Failed to initialize Gemini: {e}")
    
    def log(self, message: str, level: str = "info"):
        if level == "info":
            self.logger.info(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "warning":
            self.logger.warning(message)
    
    def track_execution(self, func):
        """Decorator for tracking agent execution metrics"""
        def wrapper(*args, **kwargs):
            start = time.time()
            metric = ExecutionMetrics(
                agent_name=self.name,
                start_time=start,
                end_time=0,
                duration=0,
                status="running"
            )
            try:
                result = func(*args, **kwargs)
                metric.status = "success"
                return result
            except Exception as e:
                metric.status = "failed"
                metric.error = str(e)
                self.log(f"Error: {e}", "error")
                raise
            finally:
                metric.end_time = time.time()
                metric.duration = metric.end_time - metric.start_time
                self.metrics.append(metric)
                self.log(f"Execution completed in {metric.duration:.2f}s - Status: {metric.status}")
        return wrapper
    
    def query_gemini(self, prompt: str, max_retries: int = 3) -> str:
        """Query Gemini with retry logic"""
        if not self.gemini_model:
            return f"[Gemini API not configured - using fallback analysis]"
        
        for attempt in range(max_retries):
            try:
                response = self.gemini_model.generate_content(prompt)
                return response.text
            except Exception as e:
                self.log(f"Gemini API error (attempt {attempt+1}): {e}", "warning")
                if attempt == max_retries - 1:
                    return f"[Gemini API unavailable - using fallback analysis]"
                time.sleep(1)

# ============================================================================
# TOOLS
# ============================================================================

class GoogleSearchTool:
    """Google Custom Search API integration"""
    def __init__(self, api_key: str, engine_id: str):
        self.api_key = api_key
        self.engine_id = engine_id
        self.logger = logging.getLogger("GoogleSearchTool")
        self.demo_mode = not api_key or "demo_mode" in api_key.lower() or not engine_id or "demo_mode" in engine_id.lower()
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """Search Google and return results"""
        try:
            if self.demo_mode:
                self.logger.info("Running in DEMO MODE - using simulated search data")
                return self._simulated_search(query)
            
            service = build("customsearch", "v1", developerKey=self.api_key)
            result = service.cse().list(q=query, cx=self.engine_id, num=num_results).execute()
            
            results = []
            for item in result.get('items', []):
                results.append({
                    'title': item.get('title'),
                    'snippet': item.get('snippet'),
                    'link': item.get('link')
                })
            return results
        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return self._simulated_search(query)
    
    def _simulated_search(self, query: str) -> List[Dict[str, str]]:
        """Fallback simulated search results"""
        knowledge_base = {
            "Global Textiles": [
                {"title": "Global Textiles - Annual Sustainability Report 2024", 
                 "snippet": "Achieved 15% reduction in carbon emissions. ISO 14001 and GOTS certified.",
                 "link": "https://example.com/global-textiles-report"}
            ],
            "EcoFabrics": [
                {"title": "EcoFabrics Wins Zero-Waste Award",
                 "snippet": "Pioneering zero-waste water treatment initiative. Fair Trade and GOTS certified.",
                 "link": "https://example.com/ecofabrics-award"}
            ],
            "Premium Weave": [
                {"title": "Premium Weave Co - Company Profile",
                 "snippet": "ISO 9001 certified. Focus on fast delivery and cost efficiency.",
                 "link": "https://example.com/premium-weave"}
            ],
            "Sustainable Threads": [
                {"title": "Sustainable Threads - Sustainability Initiatives",
                 "snippet": "OEKO-TEX certified. Renewable energy powered facilities.",
                 "link": "https://example.com/sustainable-threads"}
            ],
            "FastFabric": [
                {"title": "FastFabric Solutions - Labor Dispute Resolved",
                 "snippet": "Minor labor dispute reported 6 months ago, now resolved. Fast turnaround times.",
                 "link": "https://example.com/fastfabric-news"}
            ],
            "Quality First": [
                {"title": "Quality First Textiles Receives Top Supplier Rating",
                 "snippet": "Top tier supplier for 3 consecutive years. Full certification portfolio.",
                 "link": "https://example.com/quality-first"}
            ]
        }
        
        for vendor_name, results in knowledge_base.items():
            if vendor_name.lower() in query.lower():
                return results
        
        return [{"title": f"Search results for {query}", 
                "snippet": "Standard company information available.",
                "link": "https://example.com"}]

class GoogleSheetsTool:
    """Google Sheets API integration for persistent storage"""
    def __init__(self):
        self.logger = logging.getLogger("GoogleSheetsTool")
        self.simulated_storage = []  # Fallback for demo
    
    def append_evaluation(self, record: EvaluationRecord):
        """Append evaluation record to Google Sheets"""
        try:
            # In production, use actual Google Sheets API
            # For demo, store in memory
            self.simulated_storage.append(record)
            self.logger.info(f"Evaluation record saved: {record.id}")
        except Exception as e:
            self.logger.error(f"Failed to save to sheets: {e}")
    
    def get_vendor_history(self, vendor_id: str) -> List[Dict]:
        """Retrieve historical data for vendor"""
        history = [r for r in self.simulated_storage 
                  if any(v['id'] == vendor_id for v in r.results)]
        return history

# ============================================================================
# AGENTS
# ============================================================================

class DataCollectionAgent(BaseAgent):
    """Collects and enriches vendor data using Google Search"""
    def __init__(self, bus: MessageBus, search_tool: GoogleSearchTool, gemini_api_key: str = None):
        super().__init__("DataCollector", bus, gemini_api_key)
        self.search_tool = search_tool
    
    def run(self, vendors: List[Vendor]) -> List[Vendor]:
        @self.track_execution
        def _execute():
            self.log("Starting data collection and enrichment...")
            
            for vendor in vendors:
                # Search for vendor information
                query = f"{vendor.name} textile sustainability certifications compliance"
                search_results = self.search_tool.search(query, num_results=3)
                
                # Aggregate evidence
                evidence = []
                for result in search_results:
                    evidence.append(f"[{result['title']}] {result['snippet']}")
                
                vendor.evidence_found = " | ".join(evidence)
                
                # Update certifications if found
                evidence_text = vendor.evidence_found.lower()
                found_certs = []
                cert_keywords = ["iso 9001", "iso 14001", "gots", "fair trade", "oeko-tex"]
                for cert in cert_keywords:
                    if cert in evidence_text:
                        found_certs.append(cert.upper())
                
                if found_certs and vendor.certifications == "None":
                    vendor.certifications = ", ".join(found_certs)
                
                self.log(f"Enriched data for {vendor.name}")
                
                # Publish message to bus
                message = AgentMessage(
                    sender=self.name,
                    recipient="ESGAgent",
                    payload={"vendor_id": vendor.id, "evidence": vendor.evidence_found},
                    timestamp=time.time(),
                    message_type="data_enrichment"
                )
                self.bus.publish(message)
            
            return vendors
        
        return _execute()

class ESGAgent(BaseAgent):
    """Analyzes sustainability and ESG performance using Gemini LLM"""
    def __init__(self, bus: MessageBus, gemini_api_key: str = None):
        super().__init__("ESGAgent", bus, gemini_api_key)
    
    def run(self, vendors: List[Vendor]) -> List[Vendor]:
        @self.track_execution
        def _execute():
            self.log("Analyzing sustainability and ESG performance with Gemini AI...")
            
            for vendor in vendors:
                # Wait for data enrichment message
                messages = self.bus.get_messages_for(self.name, "data_enrichment")
                
                # Construct prompt for Gemini
                prompt = f"""
                You are an expert sustainability and ESG data analyst for textile procurement.
                
                Analyze the following vendor and provide scores (0-100) for:
                1. Carbon Footprint Management (carbon_score)
                2. Labor Practices & Ethics (labor_score)
                3. Waste & Resource Management (waste_score)
                
                Vendor: {vendor.name}
                Declared Certifications: {vendor.certifications}
                Evidence Found: {vendor.evidence_found}
                Risk Level: {vendor.supply_chain_risk}/100
                
                Provide your analysis in this exact JSON format:
                {{
                  "carbon_score": <number 0-100>,
                  "labor_score": <number 0-100>,
                  "waste_score": <number 0-100>,
                  "reasoning": "<2-3 sentences explaining your scores>"
                }}
                
                Consider:
                - ISO 14001: +environmental management
                - GOTS: +organic standards, labor, environment
                - Fair Trade: ++labor practices
                - OEKO-TEX: +textile safety, chemicals
                - Labor disputes: --labor score
                - Zero-waste initiatives: ++waste score
                """
                
                gemini_response = self.query_gemini(prompt)
                vendor.gemini_reasoning = gemini_response
                
                # Parse Gemini response (with error handling)
                try:
                    # Extract JSON from response
                    json_start = gemini_response.find('{')
                    json_end = gemini_response.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = gemini_response[json_start:json_end]
                        scores = json.loads(json_str)
                        
                        vendor.carbon_score = float(scores.get('carbon_score', 50))
                        vendor.labor_score = float(scores.get('labor_score', 50))
                        vendor.waste_score = float(scores.get('waste_score', 50))
                        vendor.audit_log = scores.get('reasoning', 'AI analysis completed')
                    else:
                        raise ValueError("No JSON found in response")
                        
                except Exception as e:
                    self.log(f"Failed to parse Gemini response for {vendor.name}: {e}", "warning")
                    # Fallback heuristic scoring
                    vendor.carbon_score, vendor.labor_score, vendor.waste_score = self._fallback_scoring(vendor)
                    vendor.audit_log = "Fallback heuristic scoring applied"
                
                # Calculate aggregate
                vendor.ESG_score = (
                    vendor.carbon_score + vendor.labor_score + vendor.waste_score
                ) / 3
                
                self.log(f"ESG score for {vendor.name}: {vendor.ESG_score:.1f}")
                
                # Publish to bus
                message = AgentMessage(
                    sender=self.name,
                    recipient="RiskAnalysisAgent",
                    payload={"vendor_id": vendor.id, "ESG_score": vendor.ESG_score},
                    timestamp=time.time(),
                    message_type="ESG_score"
                )
                self.bus.publish(message)
            
            return vendors
        
        return _execute()
    
    def _fallback_scoring(self, vendor: Vendor) -> tuple:
        """Heuristic scoring if Gemini unavailable"""
        evidence = vendor.evidence_found.lower() + " " + vendor.certifications.lower()
        c, l, w = 50.0, 50.0, 50.0
        
        if "iso 14001" in evidence: c += 20
        if "gots" in evidence: c += 10; l += 10; w += 10
        if "fair trade" in evidence: l += 30
        if "oeko-tex" in evidence: w += 20
        if "zero-waste" in evidence: w += 25; c += 15
        if "dispute" in evidence: l -= 25
        
        return (min(100, max(10, c)), min(100, max(10, l)), min(100, max(10, w)))

class RiskAnalysisAgent(BaseAgent):
    """Analyzes supply chain risks using Gemini"""
    def __init__(self, bus: MessageBus, gemini_api_key: str = None):
        super().__init__("RiskAnalysisAgent", bus, gemini_api_key)
    
    def run(self, vendors: List[Vendor]) -> List[Vendor]:
        @self.track_execution
        def _execute():
            self.log("Performing risk analysis...")
            
            for vendor in vendors:
                prompt = f"""
                You are a supply chain risk analyst for textile procurement.
                
                Analyze risks for: {vendor.name}
                - Declared Risk Score: {vendor.supply_chain_risk}/100
                - Lead Time: {vendor.lead_time} days
                - Evidence: {vendor.evidence_found[:300]}
                
                Identify and categorize risks:
                1. Delivery risk (delays, capacity)
                2. Compliance risk (certifications, audits)
                3. Reputational risk (labor disputes, scandals)
                4. Financial risk (stability)
                
                Provide a concise 2-3 sentence risk summary.
                """
                
                risk_analysis = self.query_gemini(prompt)
                vendor.risk_analysis = risk_analysis
                self.log(f"Risk analysis completed for {vendor.name}")
                
                # Publish message
                message = AgentMessage(
                    sender=self.name,
                    recipient="TOPSISAgent",
                    payload={"vendor_id": vendor.id, "risk_analysis": risk_analysis},
                    timestamp=time.time(),
                    message_type="risk_analysis"
                )
                self.bus.publish(message)
            
            return vendors
        
        return _execute()

class TOPSISRankingAgent(BaseAgent):
    """TOPSIS Multi-Criteria Decision Making"""
    def __init__(self, bus: MessageBus, gemini_api_key: str = None):
        super().__init__("TOPSISAgent", bus, gemini_api_key)
    
    def run(self, vendors: List[Vendor], weights: Dict[str, float]) -> List[Vendor]:
        @self.track_execution
        def _execute():
            self.log(f"Executing TOPSIS ranking with weights: {weights}")
            
            if not vendors:
                return []
            
            # Decision matrix
            data = np.array([
                [v.cost, v.financial_stability, v.lead_time, v.technology, 
                 v.quality, v.hygiene, v.supply_chain_risk, v.ESG_score] 
                for v in vendors
            ], dtype=float)
            
            # Normalize
            norm_data = data / np.sqrt((data**2).sum(axis=0))
            
            # Apply weights
            w_arr = np.array([
                weights['cost'], weights['financial_stability'], weights['lead_time'],
                weights['technology'], weights['quality'], weights['hygiene'], 
                weights['supply_chain_risk'], weights['ESG_score']
            ])
            weighted_data = norm_data * w_arr
            
            # Ideal solutions
            ideal_best = np.array([
                weighted_data[:, 0].min(),  # Cost (min)
                weighted_data[:, 1].max(),  # Financial Stability (max)
                weighted_data[:, 2].min(),  # Lead Time (min)
                weighted_data[:, 3].max(),  # Technology (max)
                weighted_data[:, 4].max(),  # Quality (max)
                weighted_data[:, 5].max(),  # Hygiene (max)
                weighted_data[:, 6].min(),  # Supply Chain Risk (min)
                weighted_data[:, 7].max()   # ESG Score (max)
            ])
            
            ideal_worst = np.array([
                weighted_data[:, 0].max(),
                weighted_data[:, 1].min(),
                weighted_data[:, 2].max(),
                weighted_data[:, 3].min(),
                weighted_data[:, 4].min(),
                weighted_data[:, 5].min(),
                weighted_data[:, 6].max(),
                weighted_data[:, 7].min()
            ])
            
            # Distances
            dist_best = np.sqrt(((weighted_data - ideal_best)**2).sum(axis=1))
            dist_worst = np.sqrt(((weighted_data - ideal_worst)**2).sum(axis=1))
            
            # TOPSIS scores
            scores = dist_worst / (dist_best + dist_worst + 1e-9)
            
            for i, vendor in enumerate(vendors):
                vendor.topsis_score = scores[i]
                self.log(f"TOPSIS score for {vendor.name}: {scores[i]:.7f}")
            
            # Sort by score
            vendors.sort(key=lambda x: x.topsis_score, reverse=True)
            
            # Publish results
            message = AgentMessage(
                sender=self.name,
                recipient="ValidationAgent",
                payload={"rankings": [v.id for v in vendors]},
                timestamp=time.time(),
                message_type="ranking"
            )
            self.bus.publish(message)
            
            return vendors
        
        return _execute()

class ValidationAgent(BaseAgent):
    """Validates rankings and loops if needed"""
    def __init__(self, bus: MessageBus, gemini_api_key: str = None, min_sustainability_threshold: float = 60.0):
        super().__init__("ValidationAgent", bus, gemini_api_key)
        self.min_threshold = min_sustainability_threshold
        self.max_iterations = 3
    
    def run(self, vendors: List[Vendor], weights: Dict[str, float], 
            topsis_agent: 'TOPSISRankingAgent') -> tuple:
        @self.track_execution
        def _execute():
            self.log("Validating rankings...")
            
            iteration = 0
            validation_log = []
            current_vendors = vendors
            current_weights = weights.copy()
            
            while iteration < self.max_iterations:
                iteration += 1
                top_vendor = current_vendors[0]
                
                self.log(f"Iteration {iteration}: Top vendor = {top_vendor.name}, "
                        f"ESG score = {top_vendor.ESG_score:.1f}")
                
                # Check if top vendor meets threshold
                if top_vendor.ESG_score >= self.min_threshold:
                    validation_log.append(
                        f"✓ Iteration {iteration}: {top_vendor.name} meets ESG threshold "
                        f"({top_vendor.ESG_score:.1f} >= {self.min_threshold})"
                    )
                    self.log("Validation passed!")
                    break
                else:
                    # Adjust weights to prioritize ESG
                    validation_log.append(
                        f"⚠ Iteration {iteration}: {top_vendor.name} below threshold "
                        f"({top_vendor.ESG_score:.1f} < {self.min_threshold}). "
                        f"Adjusting weights..."
                    )
                    
                    # Increase ESG score weight
                    current_weights['ESG_score'] = min(0.7, current_weights['ESG_score'] + 0.15)
                    # Redistribute other weights
                    remaining = 1.0 - current_weights['ESG_score']
                    other_keys = ['cost', 'financial_stability', 'lead_time', 'technology', 
                                  'quality', 'hygiene', 'supply_chain_risk']
                    for key in other_keys:
                        current_weights[key] = remaining / len(other_keys)
                    
                    self.log(f"Adjusted weights: {current_weights}")
                    
                    # Re-rank with new weights
                    current_vendors = topsis_agent.run(current_vendors, current_weights)
                    
                    if iteration == self.max_iterations:
                        validation_log.append(
                            f"⚠ Max iterations reached. Best achievable ESG score: "
                            f"{current_vendors[0].ESG_score:.1f}"
                        )
            
            return current_vendors, current_weights, validation_log
        
        return _execute()

class MemoryAgent(BaseAgent):
    """Manages long-term memory and historical context"""
    def __init__(self, bus: MessageBus, sheets_tool: GoogleSheetsTool, gemini_api_key: str = None):
        super().__init__("MemoryAgent", bus, gemini_api_key)
        self.sheets_tool = sheets_tool
    
    def get_vendor_context(self, vendor_id: str) -> str:
        """Retrieve and summarize vendor history"""
        @self.track_execution
        def _execute():
            history = self.sheets_tool.get_vendor_history(vendor_id)
            
            if not history:
                return "No historical data available for this vendor."
            
            # Use Gemini to summarize history
            prompt = f"""
            Summarize the evaluation history for vendor {vendor_id}:
            
            {json.dumps(history, indent=2)}
            
            Provide a concise 2-3 sentence summary highlighting:
            - Past performance trends
            - Any red flags or concerns
            - Recommendation (reliable / monitor / caution)
            """
            
            summary = self.query_gemini(prompt)
            self.log(f"Retrieved historical context for {vendor_id}")
            return summary
        
        return _execute()
    
    def save_evaluation(self, record: EvaluationRecord):
        """Persist evaluation to long-term storage"""
        @self.track_execution
        def _execute():
            self.sheets_tool.append_evaluation(record)
            self.log(f"Evaluation {record.id} saved to long-term memory")
        
        return _execute()

# ============================================================================
# ORCHESTRATOR
# ============================================================================

class MultiAgentOrchestrator:
    """Coordinates all agents and manages execution flow"""
    def __init__(self, gemini_api_key: str = None, search_api_key: str = None, search_engine_id: str = None):
        self.bus = MessageBus()
        self.search_tool = GoogleSearchTool(search_api_key or "demo_mode", search_engine_id or "demo_mode")
        self.sheets_tool = GoogleSheetsTool()
        
        # Initialize agents
        self.data_agent = DataCollectionAgent(self.bus, self.search_tool, gemini_api_key)
        self.esg_agent = ESGAgent(self.bus, gemini_api_key)
        self.risk_agent = RiskAnalysisAgent(self.bus, gemini_api_key)
        self.topsis_agent = TOPSISRankingAgent(self.bus, gemini_api_key)
        self.validation_agent = ValidationAgent(self.bus, gemini_api_key)
        self.memory_agent = MemoryAgent(self.bus, self.sheets_tool, gemini_api_key)
        
        self.logger = logging.getLogger("Orchestrator")
    
    def run_evaluation(self, vendors: List[Vendor], weights: Dict[str, float], 
                      use_parallel: bool = True) -> tuple:
        """Execute multi-agent evaluation"""
        self.logger.info(f"Starting evaluation of {len(vendors)} vendors")
        start_time = time.time()
        
        # Clear message bus
        self.bus.clear()
        
        # Phase 1: Parallel data collection and risk analysis
        if use_parallel:
            self.logger.info("Phase 1: Parallel execution (DataCollection + RiskAnalysis)")
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_data = executor.submit(self.data_agent.run, vendors)
                future_risk = executor.submit(self.risk_agent.run, vendors)
                
                vendors = future_data.result()
                vendors = future_risk.result()
        else:
            self.logger.info("Phase 1: Sequential execution")
            vendors = self.data_agent.run(vendors)
            vendors = self.risk_agent.run(vendors)
        
        # Phase 2: Sequential ESG analysis (depends on enriched data)
        self.logger.info("Phase 2: ESG analysis")
        vendors = self.esg_agent.run(vendors)
        
        # Phase 3: TOPSIS ranking
        self.logger.info("Phase 3: TOPSIS ranking")
        vendors = self.topsis_agent.run(vendors, weights)
        
        # Phase 4: Validation loop
        self.logger.info("Phase 4: Validation loop")
        vendors, final_weights, validation_log = self.validation_agent.run(
            vendors, weights, self.topsis_agent
        )
        
        # Collect metrics
        total_time = time.time() - start_time
        metrics = {
            'total_duration': total_time,
            'agent_metrics': {
                'data_collection': self.data_agent.metrics[-1].duration if self.data_agent.metrics else 0,
                'ESG_analysis': self.esg_agent.metrics[-1].duration if self.esg_agent.metrics else 0,
                'risk_analysis': self.risk_agent.metrics[-1].duration if self.risk_agent.metrics else 0,
                'topsis': self.topsis_agent.metrics[-1].duration if self.topsis_agent.metrics else 0,
                'validation': self.validation_agent.metrics[-1].duration if self.validation_agent.metrics else 0,
            },
            'validation_log': validation_log,
            'final_weights': final_weights,
            'messages_exchanged': len(self.bus.messages)
        }
        
        self.logger.info(f"Evaluation completed in {total_time:.2f}s")
        
        return vendors, metrics

# ============================================================================
# STREAMLIT UI
# ============================================================================

def init_session():
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    if 'page' not in st.session_state:
        st.session_state.page = "Login"
    
    if 'vendors' not in st.session_state:
        st.session_state.vendors = [
            Vendor("V001", "Global Textiles Ltd", 450, 95, 28, 75, 65, 70, 15, "ISO 9001, ISO 14001, GOTS"),
            Vendor("V002", "EcoFabrics Inc", 1200, 60, 35, 90, 85, 90, 40, "ISO 9001, ISO 14001, GOTS, Fair Trade"),
            Vendor("V003", "Premium Weave Co", 1550, 85, 42, 65, 98, 85, 20, "ISO 9001"),
            Vendor("V004", "Sustainable Threads", 1100, 55, 28, 70, 80, 85, 45, "ISO 9001, ISO 14001, GOTS, OEKO-TEX"),
            Vendor("V005", "FastFabric Solutions", 700, 70, 7, 95, 60, 65, 30, "None"),
            Vendor("V006", "Quality First Textiles", 850, 80, 35, 60, 95, 90, 10, "ISO 9001, ISO 14001, GOTS, Fair Trade, OEKO-TEX"),
        ]
    
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = None
    
    if 'last_results' not in st.session_state:
        st.session_state.last_results = None
    
    if 'last_metrics' not in st.session_state:
        st.session_state.last_metrics = None
    
    # API Configuration
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = None
    
    if 'search_api_key' not in st.session_state:
        st.session_state.search_api_key = None
    
    if 'search_engine_id' not in st.session_state:
        st.session_state.search_engine_id = None

def login_page():
    st.markdown("""
    <style>
    .login-header {
        text-align: center;
        padding: 20px;
    }
    .api-info {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown('<div class="login-header">', unsafe_allow_html=True)
        st.title("🏭 Sustainable Vendor Decision System")
        st.markdown("**Multi-Agent AI Platform with Google Gemini**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Login credentials
        st.subheader("🔐 Login Credentials")
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        
        st.divider()
        
        # API Configuration
        st.subheader("🔧 API Configuration")
        
        with st.expander("ℹ️ How to get API Keys (Click to expand)", expanded=False):
            st.markdown("""
            ### 🤖 Gemini API Key
            1. Visit: [Google AI Studio](https://makersuite.google.com/app/apikey)
            2. Click "Create API Key"
            3. Copy and paste below
            
            ### 🔍 Google Search API
            1. **API Key**: [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
               - Create credentials → API key
               - Enable "Custom Search API"
            2. **Search Engine ID**: [Programmable Search](https://programmablesearchengine.google.com/)
               - Create new search engine
               - Set to search entire web
               - Copy the Search Engine ID
            
            ### 💡 Demo Mode
            Leave any field empty or enter "demo_mode" to use simulated data (perfect for testing!)
            """)
        
        st.markdown('<div class="api-info">', unsafe_allow_html=True)
        gemini_key = st.text_input(
            "🤖 Gemini API Key",
            type="password",
            placeholder="AIzaSy... or leave empty for demo mode",
            help="Get from https://makersuite.google.com/app/apikey"
        )
        
        search_key = st.text_input(
            "🔍 Google Search API Key",
            type="password",
            placeholder="AIzaSy... or leave empty for demo mode",
            help="Get from Google Cloud Console"
        )
        
        search_engine = st.text_input(
            "🔎 Google Search Engine ID",
            placeholder="017576662... or leave empty for demo mode",
            help="Get from https://programmablesearchengine.google.com/"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Demo mode indicator
        demo_mode = (not gemini_key or gemini_key.lower() == "demo_mode" or 
                    not search_key or search_key.lower() == "demo_mode" or
                    not search_engine or search_engine.lower() == "demo_mode")
        
        if demo_mode:
            st.info("🎭 **Demo Mode Active** - System will use simulated data and fallback heuristics")
        else:
            st.success("✅ **Live Mode** - Real API integration enabled")
        
        st.divider()
        
        if st.button("🚀 Sign In & Initialize System", type="primary", use_container_width=True):
            if username == "CogitoCore":
                with st.spinner("Initializing multi-agent system..."):
                    # Store API keys
                    st.session_state.gemini_api_key = gemini_key if gemini_key else "demo_mode"
                    st.session_state.search_api_key = search_key if search_key else "demo_mode"
                    st.session_state.search_engine_id = search_engine if search_engine else "demo_mode"
                    
                    # Initialize orchestrator with API keys
                    st.session_state.orchestrator = MultiAgentOrchestrator(
                        gemini_api_key=st.session_state.gemini_api_key,
                        search_api_key=st.session_state.search_api_key,
                        search_engine_id=st.session_state.search_engine_id
                    )
                    
                    st.session_state.user = username
                    st.session_state.page = "Dashboard"
                    time.sleep(1)
                    st.rerun()
            else:
                st.error("❌ Invalid Username (Hint: CogitoCore)")
        
        st.caption("💡 **Quick Start**: Use username 'CogitoCore' and leave API fields empty for demo mode")

def sidebar_nav():
    with st.sidebar:
        st.title("🤖 Sustainable Vendor Decision System")
        st.markdown(f"**User:** {st.session_state.user}")
        
        # API Status
        demo_mode = "demo_mode" in str(st.session_state.gemini_api_key).lower()
        if demo_mode:
            st.warning("🎭 Demo Mode")
        else:
            st.success("✅ Live APIs")
        
        st.divider()
        
        pages = {
            "Dashboard": "📊",
            "Vendors": "👥",
            "New Evaluation": "📝",
            "History": "🕐",
            "System Metrics": "📈",
            "API Settings": "⚙️"
        }
        
        for page, icon in pages.items():
            if st.button(f"{icon} {page}", use_container_width=True):
                st.session_state.page = page
                st.rerun()
        
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.user = None
            st.session_state.page = "Login"
            st.session_state.orchestrator = None
            st.rerun()

def dashboard_page():
    st.title("🏭 Multi-Agent Vendor Decision System")
    st.markdown("**Powered by Google Gemini & Advanced Analytics**")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Vendors", len(st.session_state.vendors))
    c2.metric("Evaluations Run", len(st.session_state.history))
    c3.metric("Agents Active", "6")
    c4.metric("Messages Exchanged", 
              st.session_state.last_metrics['messages_exchanged'] 
              if st.session_state.last_metrics else 0)
    
    st.divider()
    
    # Recent evaluations
    st.subheader("📋 Recent Evaluations")
    if not st.session_state.history:
        st.info("No evaluations yet. Create a new evaluation to get started!")
    else:
        for rec in list(reversed(st.session_state.history))[:5]:
            with st.expander(f"📄 {rec.name} - {datetime.fromtimestamp(rec.timestamp).strftime('%Y-%m-%d %H:%M')}"):
                st.write(f"**Vendors:** {', '.join(rec.vendors_involved)}")
                st.write(f"**Execution Time:** {rec.execution_metrics.get('total_duration', 0):.2f}s")
                if st.button(f"View Results", key=f"view_{rec.id}"):
                    st.session_state.last_results = rec
                    st.session_state.last_metrics = rec.execution_metrics
                    st.session_state.page = "Results"
                    st.rerun()

def vendor_mgmt_page():
    st.title("👥 Vendor Management")
    
    # Initialize edit mode state
    if 'edit_vendor_id' not in st.session_state:
        st.session_state.edit_vendor_id = None
    
    tab1, tab2, tab3 = st.tabs(["📋 View Vendors", "➕ Add Vendor", "✏️ Edit Vendor"])
    
    with tab1:
        df = pd.DataFrame([asdict(v) for v in st.session_state.vendors])
        display_cols = ['id', 'name', 'cost', 'financial_stability', 'lead_time', 'technology', 
                       'quality', 'hygiene', 'supply_chain_risk', 'certifications']
        st.dataframe(df[display_cols], use_container_width=True, height=400)
        
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        # Edit functionality
        with col1:
            edit_id = st.selectbox(
                "Select Vendor to Edit",
                options=[""] + [v.id for v in st.session_state.vendors],
                key="edit_select"
            )
            if st.button("✏️ Edit", type="primary", disabled=not edit_id):
                st.session_state.edit_vendor_id = edit_id
                st.session_state.page = "Vendors"  # Stay on same page
                st.rerun()
        
        # Delete functionality
        with col2:
            del_id = st.selectbox(
                "Select Vendor to Delete",
                options=[""] + [v.id for v in st.session_state.vendors],
                key="delete_select"
            )
            if st.button("🗑️ Delete", type="secondary", disabled=not del_id):
                vendor_name = next((v.name for v in st.session_state.vendors if v.id == del_id), "")
                st.session_state.vendors = [v for v in st.session_state.vendors if v.id != del_id]
                st.success(f"✅ Deleted {vendor_name}!")
                st.rerun()
        
        # Quick stats
        with col3:
            st.metric("Total Vendors", len(st.session_state.vendors))
            avg_quality = np.mean([v.quality for v in st.session_state.vendors])
            st.metric("Avg Quality", f"{avg_quality:.1f}/100")
    
    with tab2:
        st.subheader("➕ Add New Vendor")
        with st.form("add_vendor"):
            col1, col2 = st.columns(2)
            with col1:
                v_id = st.text_input("Vendor ID*", f"V{len(st.session_state.vendors)+1:03d}")
                v_name = st.text_input("Vendor Name*")
                v_cost = st.number_input("Cost (USD) per 100 yards*", min_value=0.0, value=1000.0, step=100.0)
                v_financial_stability = st.number_input("Financial Stability (0-100)*", 0, 100, 80)
                v_lead_time = st.number_input("Lead Time (days)*", 1, 365, 30)
                v_technology = st.number_input("Technology Score (0-100)*", 0, 100, 75)  
            with col2:
                v_quality = st.number_input("Quality (0-100)*", 0, 100, 85)
                v_hygiene = st.number_input("Hygiene Score (0-100)*", 0, 100, 80)
                v_supply_chain_risk = st.number_input("Supply Chain Risk (0-100)*", 0, 100, 20)
                v_certs = st.text_input("Certifications", "None")
            
            st.caption("* Required fields")
            
            col_submit, col_reset = st.columns([1, 3])
            with col_submit:
                submitted = st.form_submit_button("➕ Add Vendor", type="primary", use_container_width=True)
            
            if submitted:
                # Validation
                if not v_name:
                    st.error("❌ Vendor name is required!")
                elif v_id in [v.id for v in st.session_state.vendors]:
                    st.error(f"❌ Vendor ID '{v_id}' already exists!")
                else:
                    new_vendor = Vendor(v_id, v_name, v_cost, v_financial_stability, v_lead_time, 
                                       v_technology, v_quality, v_hygiene, v_supply_chain_risk, v_certs)
                    st.session_state.vendors.append(new_vendor)
                    st.success(f"✅ Added {v_name} successfully!")
                    time.sleep(1)
                    st.rerun()
    
    with tab3:
        st.subheader("✏️ Edit Vendor")
        
        if st.session_state.edit_vendor_id:
            # Find the vendor to edit
            vendor_to_edit = next((v for v in st.session_state.vendors if v.id == st.session_state.edit_vendor_id), None)
            
            if vendor_to_edit:
                st.info(f"Editing: **{vendor_to_edit.name}** (ID: {vendor_to_edit.id})")
                
                with st.form("edit_vendor"):
                    col1, col2 = st.columns(2)
                    with col1:
                        e_id = st.text_input("Vendor ID", vendor_to_edit.id, disabled=True, help="ID cannot be changed")
                        e_name = st.text_input("Vendor Name*", vendor_to_edit.name)
                        e_cost = st.number_input("Cost (USD) per 100 yards*", min_value=0.0, value=float(vendor_to_edit.cost), step=100.0)
                        e_financial_stability = st.number_input("Financial Stability (0-100)*", 0, 100, int(vendor_to_edit.financial_stability))
                        e_lead_time = st.number_input("Lead Time (days)*", 1, 365, int(vendor_to_edit.lead_time))
                        e_technology = st.number_input("Technology Score (0-100)*", 0, 100, int(vendor_to_edit.technology))
                    with col2:
                        e_quality = st.number_input("Quality (0-100)*", 0, 100, int(vendor_to_edit.quality))
                        e_hygiene = st.number_input("Hygiene Score (0-100)*", 0, 100, int(vendor_to_edit.hygiene))
                        e_supply_chain_risk = st.number_input("Supply Chain Risk (0-100)*", 0, 100, int(vendor_to_edit.supply_chain_risk))
                        e_certs = st.text_input("Certifications", vendor_to_edit.certifications)
                    
                    st.caption("* Required fields")
                    
                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        save_clicked = st.form_submit_button("💾 Save Changes", type="primary", use_container_width=True)
                    with col_cancel:
                        cancel_clicked = st.form_submit_button("❌ Cancel", use_container_width=True)
                    
                    if save_clicked:
                        if not e_name:
                            st.error("❌ Vendor name is required!")
                        else:
                            # Update the vendor
                            vendor_to_edit.name = e_name
                            vendor_to_edit.cost = e_cost
                            vendor_to_edit.financial_stability = e_financial_stability
                            vendor_to_edit.lead_time = e_lead_time
                            vendor_to_edit.technology = e_technology
                            vendor_to_edit.quality = e_quality
                            vendor_to_edit.hygiene = e_hygiene
                            vendor_to_edit.supply_chain_risk = e_supply_chain_risk
                            vendor_to_edit.certifications = e_certs
                        
                            st.success(f"✅ Updated {e_name} successfully!")
                            st.session_state.edit_vendor_id = None
                            time.sleep(1)
                            st.rerun()
                    
                    if cancel_clicked:
                        st.session_state.edit_vendor_id = None
                        st.rerun()
            else:
                st.error("❌ Vendor not found!")
                st.session_state.edit_vendor_id = None
        else:
            st.info("👈 Select a vendor from the **View Vendors** tab to edit")
            
            # Show all vendors for quick reference
            st.markdown("### Available Vendors:")
            for v in st.session_state.vendors:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{v.name}** (ID: {v.id})")
                with col2:
                    if st.button("Edit", key=f"quick_edit_{v.id}", use_container_width=True):
                        st.session_state.edit_vendor_id = v.id
                        st.rerun()

def new_evaluation_page():
    st.title("📝 New Evaluation")
    
    if 'eval_step' not in st.session_state:
        st.session_state.eval_step = 1
    
    progress = st.session_state.eval_step / 3
    st.progress(progress, text=f"Step {st.session_state.eval_step} of 3")
    
    # Step 1: Select vendors
    if st.session_state.eval_step == 1:
        st.subheader("Step 1: Select Vendors")
        
        cols = st.columns(3)
        selected = []
        for i, v in enumerate(st.session_state.vendors):
            with cols[i % 3]:
                if st.checkbox(f"**{v.name}**", key=f"sel_{v.id}"):
                    selected.append(v)
                st.caption(f"💰 ${v.cost:,.0f} | ⭐ {v.quality}/100")
        
        if st.button("Next →", type="primary"):
            if not selected:
                st.warning("Select at least one vendor")
            else:
                st.session_state.eval_cart = selected
                st.session_state.eval_step = 2
                st.rerun()
    
    # Step 2: Configure weights
    elif st.session_state.eval_step == 2:
        st.subheader("Step 2: Weight Configuration")
        
        preset = st.selectbox("Preset", ["Custom", "Balanced", "Cost Focused", 
                                          "ESG-oriented", "Business Sustainability prioritized"])
        
        presets = {
            "Balanced": {"cost": 0.125, "financial_stability": 0.125, "lead_time": 0.125, "technology": 0.125, 
                        "quality": 0.125, "hygiene": 0.125, "supply_chain_risk": 0.125, "ESG_score": 0.125},
            "Cost Focused": {"cost": 0.45, "financial_stability": 0.2, "lead_time": 0.1, "technology": 0.05, 
                           "quality": 0.1, "hygiene": 0.05, "supply_chain_risk": 0.049, "ESG_score": 0.001},
            "ESG-oriented": {"cost": 0.15, "financial_stability": 0.1, "lead_time": 0.05, "technology": 0.05, 
                           "quality": 0.1, "hygiene": 0.1, "supply_chain_risk": 0.15, "ESG_score": 0.3},
            "Business Sustainability prioritized": {"cost": 0.2, "financial_stability": 0.2, "lead_time": 0.1, 
                                                   "technology": 0.1, "quality": 0.1, "hygiene": 0.1, 
                                                   "supply_chain_risk": 0.1, "ESG_score": 0.1}
        }
        
        defaults = presets.get(preset, presets["Balanced"])
        
        col1, col2 = st.columns(2)
        with col1:
            w_cost = st.slider("💰 Cost", 0.0, 1.0, defaults['cost'])
            w_financial_stability = st.slider("🏦 Financial Stability", 0.0, 1.0, defaults['financial_stability'])
            w_lead_time = st.slider("🚚 Lead Time", 0.0, 1.0, defaults['lead_time'])
            w_technology = st.slider("🛠️ Technology", 0.0, 1.0, defaults['technology'])
        with col2:
            w_quality = st.slider("⭐ Quality", 0.0, 1.0, defaults['quality'])
            w_hygiene = st.slider("🧼 Hygiene", 0.0, 1.0, defaults['hygiene'])
            w_risk = st.slider("⚠️ Supply Chain Risk", 0.0, 1.0, defaults['supply_chain_risk'])
            w_ESG = st.slider("🌱 ESG Score", 0.0, 1.0, defaults['ESG_score'])
        
        total = w_cost + w_financial_stability + w_lead_time + w_quality + w_technology + w_hygiene + w_risk + w_ESG
        st.info(f"Total weight: {total:.2f} (will be normalized)")
        
        col_b, col_n = st.columns([1, 1])
        if col_b.button("← Back"):
            st.session_state.eval_step = 1
            st.rerun()
        if col_n.button("Next →", type="primary"):
            if total == 0:
                st.error("Total weight cannot be zero")
            else:
                st.session_state.eval_weights = {
                    "cost": w_cost/total,
                    "financial_stability": w_financial_stability/total,
                    "lead_time": w_lead_time/total,
                    "technology": w_technology/total,
                    "quality": w_quality/total,
                    "hygiene": w_hygiene/total,
                    "supply_chain_risk": w_risk/total,
                    "ESG_score": w_ESG/total,
                }
                st.session_state.eval_step = 3
                st.rerun()
    
    # Step 3: Execute
    elif st.session_state.eval_step == 3:
        st.subheader("Step 3: Review & Execute")
        
        st.write(f"**Vendors:** {len(st.session_state.eval_cart)}")
        for v in st.session_state.eval_cart:
            st.markdown(f"- {v.name}")
        
        st.write("**Weights:**")
        st.json(st.session_state.eval_weights)
        
        eval_name = st.text_input("Evaluation Name", 
                                  f"Eval_{datetime.now().strftime('%Y%m%d_%H%M')}")
        
        use_parallel = st.checkbox("Enable parallel agent execution", value=True)
        
        col_b, col_e = st.columns([1, 1])
        if col_b.button("← Back"):
            st.session_state.eval_step = 2
            st.rerun()
        
        if col_e.button("🚀 Execute", type="primary"):
            if not st.session_state.orchestrator:
                st.error("❌ System not initialized. Please logout and login again.")
                return
            
            with st.spinner("🤖 Multi-agent system executing..."):
                # Show agent progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Phase 1: Data collection & enrichment...")
                progress_bar.progress(0.2)
                
                # Run orchestrator
                vendors, metrics = st.session_state.orchestrator.run_evaluation(
                    st.session_state.eval_cart.copy(),
                    st.session_state.eval_weights,
                    use_parallel=use_parallel
                )
                
                progress_bar.progress(0.5)
                status_text.text("Phase 2: ESG analysis...")
                time.sleep(0.5)
                
                progress_bar.progress(0.8)
                status_text.text("Phase 3: TOPSIS ranking & validation...")
                time.sleep(0.5)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Complete!")
                
                # Save results
                record = EvaluationRecord(
                    id=eval_name,
                    timestamp=time.time(),
                    name=eval_name,
                    vendors_involved=[v.name for v in vendors],
                    weights=st.session_state.eval_weights,
                    results=[asdict(v) for v in vendors],
                    execution_metrics=metrics
                )
                
                st.session_state.history.append(record)
                st.session_state.last_results = record
                st.session_state.last_metrics = metrics
                
                # Save to memory
                st.session_state.orchestrator.memory_agent.save_evaluation(record)
                
                st.session_state.eval_step = 1
                st.session_state.page = "Results"
                st.rerun()

def results_page():
    st.title("🏆 Evaluation Results")
    
    rec = st.session_state.last_results
    if not rec:
        st.error("No results found")
        return
    
    st.caption(f"**{rec.name}** | {datetime.fromtimestamp(rec.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Metrics summary
    metrics = rec.execution_metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⏱️ Total Time", f"{metrics.get('total_duration', 0):.2f}s")
    col2.metric("📨 Messages", metrics.get('messages_exchanged', 0))
    col3.metric("🔄 Validation Loops", len(metrics.get('validation_log', [])))
    col4.metric("👥 Vendors", len(rec.results))
    
    st.divider()
    
    # Rankings table
    st.subheader("📊 Rankings")
    df = pd.DataFrame(rec.results)
    df_display = df[['name', 'topsis_score', 'ESG_score', 'cost', 'financial_stability',
                     'technology', 'quality', 'hygiene', 'supply_chain_risk']].copy()
    df_display.columns = ['Vendor', 'TOPSIS Score', 'ESG Score', 'Cost', 
                          'Financial Stability', 'Technology', 'Quality', 'Hygiene', 'Supply Chain Risk']
    df_display['TOPSIS Score'] = df_display['TOPSIS Score'].map('{:.7f}'.format)
    df_display['ESG Score'] = df_display['ESG Score'].map('{:.1f}'.format)
    df_display.index += 1
    
    st.dataframe(df_display, use_container_width=True)
    
    # Winner highlight
    winner = rec.results[0]
    st.success(f"🥇 **Recommended Vendor:** {winner['name']} (Score: {winner['topsis_score']:.4f})")
    
    st.divider()
    
    # Detailed analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Radar", "🌱 ESG and Risk Analysis", 
                                       "📈 Agent Metrics", "🔍 Validation Log"])
    
    with tab1:
        st.subheader("Top 3 Vendor Comparison")
        cols = st.columns(3)
        categories = ['Cost', 'Financial Stability', 'Lead Time', 'Technology', 'Quality', 'Hygiene', 'Supply Chain Risk', 'ESG']
        
        for i, v_dict in enumerate(rec.results[:3]):
            vals = [
                100 - (v_dict['cost']/20),
                v_dict['financial_stability'],
                100 - (v_dict['lead_time']*2),
                v_dict['technology'],
                v_dict['quality'],
                v_dict['hygiene'],
                100 - v_dict['supply_chain_risk'],
                v_dict['ESG_score']
            ]
            vals = [max(0, min(100, x)) for x in vals]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=categories, fill='toself',
                name=v_dict['name']
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False, title=f"#{i+1} {v_dict['name']}",
                height=350
            )
            with cols[i]:
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🌱 ESG and Risk Analysis")
        for i, v_dict in enumerate(rec.results):
            with st.expander(f"#{i+1} {v_dict['name']} (Score: {v_dict['ESG_score']:.1f})"):
                c1, c2, c3 = st.columns(3)
                c1.metric("🌍 Carbon", f"{v_dict['carbon_score']:.0f}/100")
                c2.metric("👷 Labor", f"{v_dict['labor_score']:.0f}/100")
                c3.metric("♻️ Waste", f"{v_dict['waste_score']:.0f}/100")
                
                st.markdown("**🔍 Certifications:**")
                st.info(v_dict['certifications'])
                
                st.markdown("**🤖 Gemini Analysis:**")
                st.write(v_dict.get('audit_log', 'No analysis available'))
                
                if v_dict.get('risk_analysis'):
                    st.markdown("**⚠️ Risk Analysis:**")
                    st.warning(v_dict['risk_analysis'])
    
    with tab3:
        st.subheader("📈 Agent Execution Metrics")
        agent_metrics = metrics.get('agent_metrics', {})
        
        if agent_metrics:
            # Bar chart
            fig = px.bar(
                x=list(agent_metrics.keys()),
                y=list(agent_metrics.values()),
                labels={'x': 'Agent', 'y': 'Duration (seconds)'},
                title='Agent Execution Time'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Table
            st.dataframe(pd.DataFrame([
                {'Agent': k, 'Duration (s)': f"{v:.3f}"} 
                for k, v in agent_metrics.items()
            ]), use_container_width=True)
    
    with tab4:
        st.subheader("🔍 Validation Loop Log")
        validation_log = metrics.get('validation_log', [])
        if validation_log:
            for entry in validation_log:
                if '✓' in entry:
                    st.success(entry)
                elif '⚠' in entry:
                    st.warning(entry)
                else:
                    st.info(entry)
        else:
            st.info("No validation log available")
        
        st.markdown("**Final Weights:**")
        final_weights = metrics.get('final_weights', rec.weights)
        st.json(final_weights)
    
    # Export
    st.divider()
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Results (CSV)",
        data=csv,
        file_name=f"{rec.id}_results.csv",
        mime="text/csv"
    )
    
    if st.button("← Back to Dashboard"):
        st.session_state.page = "Dashboard"
        st.rerun()

def history_page():
    st.title("🕐 Evaluation History")
    
    if not st.session_state.history:
        st.info("No evaluation history yet")
        return
    
    for i, rec in enumerate(reversed(st.session_state.history)):
        with st.expander(f"📄 {rec.name} - {datetime.fromtimestamp(rec.timestamp).strftime('%Y-%m-%d %H:%M')}"):
            col1, col2, col3 = st.columns(3)
            col1.write(f"**Vendors:** {len(rec.results)}")
            col2.write(f"**Duration:** {rec.execution_metrics.get('total_duration', 0):.2f}s")
            col3.write(f"**Messages:** {rec.execution_metrics.get('messages_exchanged', 0)}")
            
            st.write("**Top 3:**")
            for j, v in enumerate(rec.results[:3]):
                st.markdown(f"{j+1}. **{v['name']}** (Score: {v['topsis_score']:.7f})")
            
            if st.button(f"View Full Results", key=f"hist_{i}"):
                st.session_state.last_results = rec
                st.session_state.last_metrics = rec.execution_metrics
                st.session_state.page = "Results"
                st.rerun()

def system_metrics_page():
    st.title("📈 System Metrics & Observability")
    
    if not st.session_state.orchestrator:
        st.warning("System not initialized. Please login first.")
        return
    
    st.subheader("🤖 Agent Status")
    orchestrator = st.session_state.orchestrator
    
    agents = [
        ("DataCollector", orchestrator.data_agent),
        ("ESGAgent", orchestrator.esg_agent),
        ("RiskAnalysisAgent", orchestrator.risk_agent),
        ("TOPSISAgent", orchestrator.topsis_agent),
        ("ValidationAgent", orchestrator.validation_agent),
        ("MemoryAgent", orchestrator.memory_agent)
    ]
    
    agent_data = []
    for name, agent in agents:
        total_runs = len(agent.metrics)
        avg_duration = np.mean([m.duration for m in agent.metrics]) if agent.metrics else 0
        success_rate = (
            sum(1 for m in agent.metrics if m.status == "success") / total_runs * 100 
            if total_runs > 0 else 0
        )
        
        agent_data.append({
            'Agent': name,
            'Total Runs': total_runs,
            'Avg Duration (s)': f"{avg_duration:.3f}",
            'Success Rate': f"{success_rate:.1f}%"
        })
    
    st.dataframe(pd.DataFrame(agent_data), use_container_width=True)
    
    st.divider()
    
    st.subheader("📨 Message Bus Activity")
    st.metric("Total Messages", len(orchestrator.bus.messages))
    
    if orchestrator.bus.messages:
        msg_df = pd.DataFrame([
            {
                'From': m.sender,
                'To': m.recipient,
                'Type': m.message_type,
                'Timestamp': datetime.fromtimestamp(m.timestamp).strftime('%H:%M:%S')
            }
            for m in orchestrator.bus.messages[-20:]  # Last 20 messages
        ])
        st.dataframe(msg_df, use_container_width=True)
    
    st.divider()
    
    st.subheader("📊 Performance Trends")
    if st.session_state.history:
        durations = [rec.execution_metrics.get('total_duration', 0) 
                    for rec in st.session_state.history]
        timestamps = [datetime.fromtimestamp(rec.timestamp) 
                     for rec in st.session_state.history]
        
        fig = px.line(
            x=timestamps, y=durations,
            labels={'x': 'Evaluation Time', 'y': 'Duration (seconds)'},
            title='Evaluation Performance Over Time'
        )
        st.plotly_chart(fig, use_container_width=True)

def api_settings_page():
    st.title("⚙️ API Settings")
    
    st.markdown("""
    ### Current Configuration
    View and update your API credentials. System will reinitialize on save.
    """)
    
    st.divider()
    
    # Show current status
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🤖 Gemini API")
        gemini_status = "✅ Configured" if st.session_state.gemini_api_key and "demo_mode" not in st.session_state.gemini_api_key.lower() else "🎭 Demo Mode"
        st.info(gemini_status)
    
    with col2:
        st.subheader("🔍 Google Search API")
        search_status = "✅ Configured" if st.session_state.search_api_key and "demo_mode" not in st.session_state.search_api_key.lower() else "🎭 Demo Mode"
        st.info(search_status)
    
    st.divider()
    
    # Update form
    st.subheader("🔄 Update Credentials")
    
    with st.form("update_api"):
        new_gemini = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Enter new key or leave empty for demo mode"
        )
        
        new_search = st.text_input(
            "Google Search API Key",
            type="password",
            placeholder="Enter new key or leave empty for demo mode"
        )
        
        new_engine = st.text_input(
            "Search Engine ID",
            placeholder="Enter new ID or leave empty for demo mode"
        )
        
        if st.form_submit_button("💾 Save & Reinitialize System", type="primary"):
            with st.spinner("Reinitializing system..."):
                st.session_state.gemini_api_key = new_gemini if new_gemini else "demo_mode"
                st.session_state.search_api_key = new_search if new_search else "demo_mode"
                st.session_state.search_engine_id = new_engine if new_engine else "demo_mode"
                
                # Reinitialize orchestrator
                st.session_state.orchestrator = MultiAgentOrchestrator(
                    gemini_api_key=st.session_state.gemini_api_key,
                    search_api_key=st.session_state.search_api_key,
                    search_engine_id=st.session_state.search_engine_id
                )
                
                time.sleep(1)
                st.success("✅ System reinitialized successfully!")
                st.rerun()
    
    st.divider()
    
    st.subheader("📚 Quick Links")
    st.markdown("""
    - [Gemini API Keys](https://makersuite.google.com/app/apikey)
    - [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
    - [Programmable Search Engine](https://programmablesearchengine.google.com/)
    - [API Documentation](https://cloud.google.com/apis)
    """)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Sustainable Vendor Decision System (CogitoCore)",
        page_icon="🏭",
        layout="wide"
    )
    
    init_session()
    
    if st.session_state.user is None:
        login_page()
    else:
        sidebar_nav()
        
        if st.session_state.page == "Dashboard":
            dashboard_page()
        elif st.session_state.page == "Vendors":
            vendor_mgmt_page()
        elif st.session_state.page == "New Evaluation":
            new_evaluation_page()
        elif st.session_state.page == "Results":
            results_page()
        elif st.session_state.page == "History":
            history_page()
        elif st.session_state.page == "System Metrics":
            system_metrics_page()
        elif st.session_state.page == "API Settings":
            api_settings_page()

if __name__ == "__main__":
    main()
