#!/usr/bin/env python3
"""
Investment Analysis Crew using CrewAI
Analyzes company transcripts and provides investment recommendations
"""

import os
import sys
import logging
from typing import List, Dict, Optional, Any
import pandas as pd
from pathlib import Path
import json
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from crewai import Agent, Task, Crew, Process
    from crewai_tools import BaseTool
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_community.llms import Ollama
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field
    import requests
except ImportError as e:
    logger.error(f"Missing required packages: {e}")
    print("Please install required packages:")
    print("!pip install crewai crewai-tools langchain-community langchain-openai requests pandas")
    sys.exit(1)

class InvestmentAnalysisConfig:
    """Configuration class for the investment analysis system"""
    
    def __init__(self):
        self.base_path = "/content/drive/MyDrive/Earnings2Insights/ECTsum"
        self.output_dir = "analysis_outputs"
        self.use_openai = True  # Set to False to use Ollama
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        
    def setup_llm(self):
        """Setup the language model"""
        if self.use_openai and self.openai_api_key:
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,
                api_key=self.openai_api_key
            )
        else:
            try:
                return Ollama(model="mistral")
            except Exception as e:
                logger.warning(f"Failed to setup Ollama: {e}")
                logger.info("Falling back to OpenAI. Please set OPENAI_API_KEY environment variable.")
                if self.openai_api_key:
                    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
                else:
                    raise ValueError("No valid LLM configuration found")

# Custom Tools
class ReadMarkdownFileInput(BaseModel):
    file_path: str = Field(..., description="Path to the markdown file")

class AnalyzeSentimentInput(BaseModel):
    text: str = Field(..., description="Text to analyze for sentiment")

class GetStockDataInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol to get data for")

class ReadMarkdownFileTool(BaseTool):
    name: str = "read_markdown_file"
    description: str = "Read and extract content from a markdown file"
    
    def _run(self, file_path: str) -> str:
        """Read markdown file content"""
        try:
            path = Path(file_path)
            if not path.exists():
                return f"Error: File {file_path} not found"
            
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Basic validation
            if len(content.strip()) == 0:
                return f"Warning: File {file_path} is empty"
                
            return content[:10000]  # Limit content size
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return f"Error reading file: {str(e)}"

class EnhancedSentimentTool(BaseTool):
    name: str = "analyze_sentiment"
    description: str = "Analyze sentiment and extract key financial indicators from text"
    
    def _run(self, text: str) -> str:
        """Enhanced sentiment analysis with financial context"""
        try:
            # Financial positive indicators
            positive_words = [
                'growth', 'profit', 'increase', 'strong', 'success', 'revenue',
                'expansion', 'bullish', 'outperform', 'beat', 'exceed', 'positive',
                'improvement', 'momentum', 'opportunity', 'optimistic'
            ]
            
            # Financial negative indicators
            negative_words = [
                'loss', 'decline', 'risk', 'weak', 'challenge', 'decrease',
                'bearish', 'underperform', 'miss', 'concern', 'negative',
                'uncertainty', 'volatility', 'pressure', 'headwind'
            ]
            
            text_lower = text.lower()
            positive_count = sum(text_lower.count(word) for word in positive_words)
            negative_count = sum(text_lower.count(word) for word in negative_words)
            
            # Extract key metrics mentions
            revenue_mentions = len(re.findall(r'revenue|sales|income', text_lower))
            profit_mentions = len(re.findall(r'profit|earnings|ebitda', text_lower))
            growth_mentions = len(re.findall(r'growth|expand|increase', text_lower))
            
            total_words = max(1, positive_count + negative_count)
            sentiment_score = (positive_count - negative_count) / total_words
            
            analysis = {
                'sentiment_score': round(sentiment_score, 3),
                'positive_indicators': positive_count,
                'negative_indicators': negative_count,
                'revenue_mentions': revenue_mentions,
                'profit_mentions': profit_mentions,
                'growth_mentions': growth_mentions,
                'overall_tone': 'positive' if sentiment_score > 0.1 else 'negative' if sentiment_score < -0.1 else 'neutral'
            }
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return json.dumps({'error': str(e), 'sentiment_score': 0})

class StockDataTool(BaseTool):
    name: str = "get_stock_data"
    description: str = "Get basic stock information and financial data"
    
    def __init__(self, alpha_vantage_key: Optional[str] = None):
        super().__init__()
        self.alpha_vantage_key = alpha_vantage_key
    
    def _run(self, symbol: str) -> str:
        """Get stock data using Alpha Vantage or fallback methods"""
        try:
            if self.alpha_vantage_key:
                return self._get_alpha_vantage_data(symbol)
            else:
                return self._get_basic_stock_info(symbol)
        except Exception as e:
            logger.error(f"Error getting stock data for {symbol}: {e}")
            return f"Error retrieving stock data for {symbol}: {str(e)}"
    
    def _get_alpha_vantage_data(self, symbol: str) -> str:
        """Get data from Alpha Vantage API"""
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'Error Message' in data:
                return f"Error: {data['Error Message']}"
            
            # Extract key financial metrics
            key_metrics = {
                'Symbol': data.get('Symbol', 'N/A'),
                'Name': data.get('Name', 'N/A'),
                'Sector': data.get('Sector', 'N/A'),
                'Market Cap': data.get('MarketCapitalization', 'N/A'),
                'P/E Ratio': data.get('PERatio', 'N/A'),
                'Dividend Yield': data.get('DividendYield', 'N/A'),
                'EPS': data.get('EPS', 'N/A'),
                'Revenue TTM': data.get('RevenueTTM', 'N/A'),
                'Profit Margin': data.get('ProfitMargin', 'N/A')
            }
            
            return json.dumps(key_metrics, indent=2)
            
        except Exception as e:
            return f"Error with Alpha Vantage API: {str(e)}"
    
    def _get_basic_stock_info(self, symbol: str) -> str:
        """Fallback method for basic stock information"""
        return json.dumps({
            'symbol': symbol,
            'note': 'Limited data available without API key',
            'recommendation': 'Set ALPHA_VANTAGE_API_KEY for detailed financial data'
        }, indent=2)

class InvestmentAnalysisCrew:
    """Main class for running investment analysis using CrewAI"""
    
    def __init__(self, config: InvestmentAnalysisConfig):
        self.config = config
        self.llm = config.setup_llm()
        self.tools = self._setup_tools()
        
    def _setup_tools(self):
        """Setup all tools for the crew"""
        tools = {
            'read_md': ReadMarkdownFileTool(),
            'sentiment': EnhancedSentimentTool(),
            'search': DuckDuckGoSearchRun(),
            'stock_data': StockDataTool(self.config.alpha_vantage_key)
        }
        return tools
    
    def create_transcript_analyzer_agent(self) -> Agent:
        """Create transcript analysis agent"""
        return Agent(
            role="Senior Transcript Analyst",
            goal="Extract and analyze key information from company earnings transcripts",
            backstory="""You are an expert financial analyst with 10+ years of experience 
            analyzing earnings calls and company communications. You excel at identifying 
            key business insights, management sentiment, and forward-looking statements.""",
            tools=[self.tools['read_md'], self.tools['sentiment']],
            verbose=True,
            llm=self.llm,
            allow_delegation=False,
            max_iter=3
        )
    
    def create_financial_analyst_agent(self) -> Agent:
        """Create financial analysis agent"""
        return Agent(
            role="Senior Financial Analyst",
            goal="Analyze financial health, performance metrics, and market position",
            backstory="""You are a seasoned financial analyst with 15+ years at top-tier 
            investment banks. You specialize in financial statement analysis, valuation, 
            and identifying investment opportunities and risks.""",
            tools=[self.tools['search'], self.tools['stock_data']],
            verbose=True,
            llm=self.llm,
            allow_delegation=False,
            max_iter=3
        )
    
    def create_risk_analyst_agent(self) -> Agent:
        """Create risk analysis agent"""
        return Agent(
            role="Chief Risk Officer",
            goal="Identify and assess potential risks in company operations and market position",
            backstory="""You are a former risk manager at a major financial institution 
            with expertise in credit risk, market risk, operational risk, and ESG factors. 
            You have a keen eye for identifying potential red flags and risk mitigation strategies.""",
            tools=[self.tools['search']],
            verbose=True,
            llm=self.llm,
            allow_delegation=False,
            max_iter=3
        )
    
    def create_investment_advisor_agent(self) -> Agent:
        """Create investment recommendation agent"""
        return Agent(
            role="Senior Investment Advisor",
            goal="Synthesize analysis and provide clear, actionable investment recommendations",
            backstory="""You are a seasoned investment professional with a proven track 
            record of successful stock picks and portfolio management for institutional 
            and high-net-worth clients. You excel at synthesizing complex analysis into 
            clear, actionable investment advice.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False,
            max_iter=3
        )
    
    def create_tasks(self, company_name: str, company_path: str) -> List[Task]:
        """Create analysis tasks for a company"""
        source_md_path = os.path.join(company_path, "source", "source.md")
        
        tasks = []
        
        # Task 1: Transcript Analysis
        transcript_task = Task(
            description=f"""Analyze the earnings transcript for {company_name} located at {source_md_path}.
            
            Your analysis should include:
            1. Key business highlights and achievements
            2. Management commentary on performance
            3. Growth strategies and initiatives
            4. Challenges and concerns mentioned
            5. Forward-looking statements and guidance
            6. Sentiment analysis of management tone
            
            Provide a comprehensive summary with specific quotes and data points where relevant.""",
            agent=self.create_transcript_analyzer_agent(),
            expected_output="""A detailed transcript analysis report containing:
            - Executive summary of key points
            - Business performance highlights
            - Strategic initiatives and outlook
            - Risk factors and challenges
            - Management sentiment analysis
            - Key quotes and data points""",
            output_file=os.path.join(company_path, "transcript_analysis.md")
        )
        
        # Task 2: Financial Analysis
        financial_task = Task(
            description=f"""Conduct a comprehensive financial analysis of {company_name} based on the 
            transcript analysis and available financial data.
            
            Focus on:
            1. Revenue trends and growth drivers
            2. Profitability and margin analysis
            3. Cash flow and capital allocation
            4. Key financial ratios and metrics
            5. Competitive positioning
            6. Valuation considerations
            
            Use available tools to gather additional financial data and market information.""",
            agent=self.create_financial_analyst_agent(),
            expected_output="""A comprehensive financial analysis report including:
            - Revenue and profitability trends
            - Key financial metrics and ratios
            - Capital allocation strategy
            - Competitive position assessment
            - Valuation analysis
            - Growth prospects evaluation""",
            context=[transcript_task],
            output_file=os.path.join(company_path, "financial_analysis.md")
        )
        
        # Task 3: Risk Assessment
        risk_task = Task(
            description=f"""Conduct a thorough risk assessment for {company_name} based on the 
            transcript and financial analysis.
            
            Evaluate:
            1. Industry and market risks
            2. Competitive threats
            3. Operational and execution risks
            4. Financial and credit risks
            5. Regulatory and compliance risks
            6. ESG and reputational risks
            
            Assess risk severity and potential impact on investment returns.""",
            agent=self.create_risk_analyst_agent(),
            expected_output="""A detailed risk assessment report containing:
            - Identification of key risk factors
            - Risk severity assessment (High/Medium/Low)
            - Potential impact on business and stock performance
            - Risk mitigation factors and management strategies
            - Overall risk profile summary""",
            context=[transcript_task, financial_task],
            output_file=os.path.join(company_path, "risk_assessment.md")
        )
        
        # Task 4: Investment Recommendation
        recommendation_task = Task(
            description=f"""Based on all previous analysis, provide a clear investment recommendation 
            for {company_name}.
            
            Your recommendation should:
            1. Provide a clear Buy/Hold/Sell recommendation
            2. Include a target price range if possible
            3. Specify investment time horizon
            4. List 3-5 key supporting arguments
            5. Identify main risks to the thesis
            6. Suggest position sizing considerations
            
            Make the recommendation actionable for investors.""",
            agent=self.create_investment_advisor_agent(),
            expected_output="""A final investment recommendation report including:
            - Clear recommendation (Buy/Hold/Sell)
            - Target price range and time horizon
            - Key investment thesis points
            - Main risks to consider
            - Position sizing suggestions
            - Executive summary for quick decision making""",
            context=[transcript_task, financial_task, risk_task],
            output_file=os.path.join(company_path, "investment_recommendation.md")
        )
        
        return [transcript_task, financial_task, risk_task, recommendation_task]
    
    def analyze_company(self, company_name: str, company_path: str) -> Dict[str, Any]:
        """Analyze a single company"""
        try:
            logger.info(f"Starting analysis for {company_name}")
            
            # Check if source file exists
            source_md_path = os.path.join(company_path, "source", "source.md")
            if not os.path.exists(source_md_path):
                return {
                    'company': company_name,
                    'status': 'error',
                    'message': f'Source file not found: {source_md_path}'
                }
            
            # Create output directory
            os.makedirs(company_path, exist_ok=True)
            
            # Create tasks
            tasks = self.create_tasks(company_name, company_path)
            
            # Create crew
            crew = Crew(
                agents=[
                    self.create_transcript_analyzer_agent(),
                    self.create_financial_analyst_agent(),
                    self.create_risk_analyst_agent(),
                    self.create_investment_advisor_agent()
                ],
                tasks=tasks,
                verbose=2,
                process=Process.sequential,
                memory=True
            )
            
            # Execute analysis
            result = crew.kickoff()
            
            return {
                'company': company_name,
                'status': 'success',
                'result': str(result),
                'output_files': [task.output_file for task in tasks if hasattr(task, 'output_file')]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {company_name}: {e}")
            return {
                'company': company_name,
                'status': 'error',
                'message': str(e)
            }
    
    def analyze_all_companies(self) -> List[Dict[str, Any]]:
        """Analyze all companies in the dataset"""
        results = []
        
        if not os.path.exists(self.config.base_path):
            logger.error(f"Base path not found: {self.config.base_path}")
            return results
        
        # Get all company directories
        company_dirs = [d for d in os.listdir(self.config.base_path)
                       if os.path.isdir(os.path.join(self.config.base_path, d))]
        
        logger.info(f"Found {len(company_dirs)} companies to analyze")
        
        for company in company_dirs:
            company_path = os.path.join(self.config.base_path, company)
            result = self.analyze_company(company, company_path)
            results.append(result)
            
            # Print progress
            print(f"\nCompleted analysis for {company}")
            print(f"Status: {result['status']}")
            if result['status'] == 'error':
                print(f"Error: {result['message']}")
            print("="*80)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]]) -> str:
        """Save analysis results to CSV"""
        try:
            df = pd.DataFrame(results)
            output_path = os.path.join(self.config.base_path, "analysis_results.csv")
            df.to_csv(output_path, index=False)
            
            # Create summary
            success_count = sum(1 for r in results if r['status'] == 'success')
            total_count = len(results)
            
            summary = {
                'total_companies': total_count,
                'successful_analyses': success_count,
                'failed_analyses': total_count - success_count,
                'success_rate': f"{(success_count/total_count)*100:.1f}%" if total_count > 0 else "0%"
            }
            
            summary_path = os.path.join(self.config.base_path, "analysis_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Results saved to {output_path}")
            logger.info(f"Summary saved to {summary_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return ""

def main():
    """Main function to run the investment analysis"""
    print("Investment Analysis Crew - Starting Analysis")
    print("="*80)
    
    # Setup configuration
    config = InvestmentAnalysisConfig()
    
    # Initialize the crew system
    try:
        analysis_crew = InvestmentAnalysisCrew(config)
        
        # Run analysis on all companies
        results = analysis_crew.analyze_all_companies()
        
        # Save results
        output_path = analysis_crew.save_results(results)
        
        # Print final summary
        success_count = sum(1 for r in results if r['status'] == 'success')
        total_count = len(results)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Total companies processed: {total_count}")
        print(f"Successful analyses: {success_count}")
        print(f"Failed analyses: {total_count - success_count}")
        print(f"Success rate: {(success_count/total_count)*100:.1f}%" if total_count > 0 else "0%")
        
        if output_path:
            print(f"Results saved to: {output_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    # For Google Colab, you might want to set environment variables here
    # os.environ['OPENAI_API_KEY'] = 'your-openai-key'
    # os.environ['ALPHA_VANTAGE_API_KEY'] = 'your-alpha-vantage-key'
    
    main()