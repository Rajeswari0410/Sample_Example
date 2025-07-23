#!/usr/bin/env python3
"""
Simplified Investment Analysis with CrewAI
Focuses on transcript analysis without external dependencies
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
    from langchain_community.llms import Ollama
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field
except ImportError as e:
    logger.error(f"Missing required packages: {e}")
    print("Please install required packages:")
    print("!pip install crewai crewai-tools langchain-community langchain-openai pandas")
    sys.exit(1)

class InvestmentAnalysisConfig:
    """Configuration class for the investment analysis system"""
    
    def __init__(self):
        self.base_path = "/content/drive/MyDrive/Earnings2Insights/ECTsum"
        self.use_openai = bool(os.getenv('OPENAI_API_KEY'))
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
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
                if self.openai_api_key:
                    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
                else:
                    raise ValueError("No valid LLM configuration found. Please set OPENAI_API_KEY.")

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
                
            if len(content.strip()) == 0:
                return f"Warning: File {file_path} is empty"
                
            return content[:15000]  # Limit content size
            
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
                'improvement', 'momentum', 'opportunity', 'optimistic', 'robust',
                'solid', 'healthy', 'accelerate', 'outpace'
            ]
            
            # Financial negative indicators
            negative_words = [
                'loss', 'decline', 'risk', 'weak', 'challenge', 'decrease',
                'bearish', 'underperform', 'miss', 'concern', 'negative',
                'uncertainty', 'volatility', 'pressure', 'headwind', 'struggle',
                'difficult', 'slowdown', 'deteriorate', 'weaken'
            ]
            
            text_lower = text.lower()
            positive_count = sum(text_lower.count(word) for word in positive_words)
            negative_count = sum(text_lower.count(word) for word in negative_words)
            
            # Extract key metrics mentions
            revenue_mentions = len(re.findall(r'revenue|sales|income', text_lower))
            profit_mentions = len(re.findall(r'profit|earnings|ebitda', text_lower))
            growth_mentions = len(re.findall(r'growth|expand|increase', text_lower))
            guidance_mentions = len(re.findall(r'guidance|outlook|forecast', text_lower))
            
            total_words = max(1, positive_count + negative_count)
            sentiment_score = (positive_count - negative_count) / total_words
            
            analysis = {
                'sentiment_score': round(sentiment_score, 3),
                'positive_indicators': positive_count,
                'negative_indicators': negative_count,
                'revenue_mentions': revenue_mentions,
                'profit_mentions': profit_mentions,
                'growth_mentions': growth_mentions,
                'guidance_mentions': guidance_mentions,
                'overall_tone': 'positive' if sentiment_score > 0.1 else 'negative' if sentiment_score < -0.1 else 'neutral'
            }
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return json.dumps({'error': str(e), 'sentiment_score': 0})

class FinancialMetricsExtractorTool(BaseTool):
    name: str = "extract_financial_metrics"
    description: str = "Extract and analyze financial metrics from transcript text"
    
    def _run(self, text: str) -> str:
        """Extract key financial metrics from transcript text"""
        try:
            text_lower = text.lower()
            
            # Extract revenue information with various formats
            revenue_patterns = [
                r'revenue[:\s]+\$?([0-9,.]+)\s*(million|billion|thousand|m|b|k)',
                r'sales[:\s]+\$?([0-9,.]+)\s*(million|billion|thousand|m|b|k)',
                r'total revenue[:\s]+\$?([0-9,.]+)\s*(million|billion|thousand|m|b|k)',
                r'\$([0-9,.]+)\s*(million|billion|thousand|m|b|k)\s+(?:in\s+)?(?:revenue|sales)'
            ]
            
            # Extract EPS information
            eps_patterns = [
                r'earnings per share[:\s]+\$?([0-9,.]+)',
                r'eps[:\s]+\$?([0-9,.]+)',
                r'diluted eps[:\s]+\$?([0-9,.]+)',
                r'per diluted share[:\s]+\$?([0-9,.]+)'
            ]
            
            # Extract growth information
            growth_patterns = [
                r'([0-9]+(?:\.[0-9]+)?)%\s*(?:yoy|year.over.year|growth)',
                r'(?:up|increase|growth)[:\s]+([0-9]+(?:\.[0-9]+)?)%',
                r'([0-9]+(?:\.[0-9]+)?)%\s*(?:increase|growth|higher)',
                r'grew\s+([0-9]+(?:\.[0-9]+)?)%',
                r'increased\s+([0-9]+(?:\.[0-9]+)?)%'
            ]
            
            # Extract margin information
            margin_patterns = [
                r'margin[:\s]+([0-9,.]+)%',
                r'operating margin[:\s]+([0-9,.]+)%',
                r'gross margin[:\s]+([0-9,.]+)%',
                r'profit margin[:\s]+([0-9,.]+)%'
            ]
            
            # Find all matches
            revenue_matches = []
            for pattern in revenue_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                revenue_matches.extend(matches)
            
            eps_matches = []
            for pattern in eps_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                eps_matches.extend(matches)
            
            growth_matches = []
            for pattern in growth_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                growth_matches.extend(matches)
            
            margin_matches = []
            for pattern in margin_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                margin_matches.extend(matches)
            
            # Extract key business metrics
            capex_mentions = len(re.findall(r'capex|capital expenditure|capital spending', text_lower))
            dividend_mentions = len(re.findall(r'dividend|payout', text_lower))
            acquisition_mentions = len(re.findall(r'acquisition|merger|acquire|bought', text_lower))
            debt_mentions = len(re.findall(r'debt|leverage|borrowing', text_lower))
            
            # Extract guidance keywords
            guidance_keywords = ['guidance', 'outlook', 'forecast', 'expect', 'anticipate', 'project']
            guidance_mentions = sum(text_lower.count(word) for word in guidance_keywords)
            
            # Compile results
            metrics = {
                'revenue_mentions': len(revenue_matches),
                'revenue_figures': revenue_matches[:5] if revenue_matches else [],
                'eps_mentions': len(eps_matches),
                'eps_figures': eps_matches[:3] if eps_matches else [],
                'growth_percentages': growth_matches[:5] if growth_matches else [],
                'margin_figures': margin_matches[:3] if margin_matches else [],
                'guidance_mentions': guidance_mentions,
                'capex_mentions': capex_mentions,
                'dividend_mentions': dividend_mentions,
                'acquisition_mentions': acquisition_mentions,
                'debt_mentions': debt_mentions,
                'total_metrics_found': len(revenue_matches) + len(eps_matches) + len(growth_matches) + len(margin_matches)
            }
            
            return json.dumps(metrics, indent=2)
            
        except Exception as e:
            logger.error(f"Error extracting financial metrics: {e}")
            return json.dumps({'error': str(e), 'metrics_found': 0})

class SimpleInvestmentAnalysisCrew:
    """Simplified investment analysis using CrewAI"""
    
    def __init__(self, config: InvestmentAnalysisConfig):
        self.config = config
        self.llm = config.setup_llm()
        self.tools = self._setup_tools()
        
    def _setup_tools(self):
        """Setup all tools for the crew"""
        return {
            'read_md': ReadMarkdownFileTool(),
            'sentiment': EnhancedSentimentTool(),
            'financial_metrics': FinancialMetricsExtractorTool()
        }
    
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
            max_iter=2
        )
    
    def create_financial_analyst_agent(self) -> Agent:
        """Create financial analysis agent"""
        return Agent(
            role="Senior Financial Analyst",
            goal="Analyze financial health and performance metrics from transcript data",
            backstory="""You are a seasoned financial analyst with 15+ years at top-tier 
            investment banks. You specialize in extracting and analyzing financial metrics 
            from earnings transcripts and identifying investment opportunities.""",
            tools=[self.tools['financial_metrics']],
            verbose=True,
            llm=self.llm,
            allow_delegation=False,
            max_iter=2
        )
    
    def create_investment_advisor_agent(self) -> Agent:
        """Create investment recommendation agent"""
        return Agent(
            role="Senior Investment Advisor",
            goal="Provide clear, actionable investment recommendations based on transcript analysis",
            backstory="""You are a seasoned investment professional with a proven track 
            record of successful stock picks and portfolio management. You excel at 
            synthesizing transcript analysis into clear, actionable investment advice.""",
            tools=[],
            verbose=True,
            llm=self.llm,
            allow_delegation=False,
            max_iter=2
        )
    
    def analyze_single_company(self, company_name: str) -> Dict[str, Any]:
        """Analyze a single company with simplified workflow"""
        try:
            company_path = os.path.join(self.config.base_path, company_name)
            source_md_path = os.path.join(company_path, "source", "source.md")
            
            if not os.path.exists(source_md_path):
                return {
                    'company': company_name,
                    'status': 'error',
                    'message': f'Source file not found: {source_md_path}'
                }
            
            # Create output directory
            os.makedirs(company_path, exist_ok=True)
            
            # Task 1: Transcript Analysis
            transcript_task = Task(
                description=f"""Analyze the earnings transcript for {company_name} located at {source_md_path}.
                
                Your analysis should include:
                1. Key business highlights and achievements mentioned
                2. Management commentary on performance and strategy
                3. Growth initiatives and strategic priorities
                4. Challenges, risks, and concerns discussed
                5. Forward-looking statements and guidance
                6. Overall management sentiment and tone
                
                Use the sentiment analysis tool to assess the overall tone of management commentary.
                Provide specific quotes and data points where relevant.""",
                agent=self.create_transcript_analyzer_agent(),
                expected_output="""A comprehensive transcript analysis report containing:
                - Executive summary of key points
                - Business performance highlights
                - Strategic initiatives and growth plans
                - Risk factors and challenges identified
                - Management sentiment analysis results
                - Key quotes and specific data points mentioned""",
                output_file=os.path.join(company_path, "transcript_analysis.md")
            )
            
            # Task 2: Financial Metrics Extraction and Analysis
            financial_task = Task(
                description=f"""Extract and analyze financial metrics from the {company_name} transcript.
                
                Focus on:
                1. Use the financial metrics extraction tool to identify specific numbers
                2. Analyze revenue figures, growth rates, and trends
                3. Extract EPS data and profitability metrics
                4. Identify margin information and operational efficiency
                5. Assess guidance and forward-looking financial statements
                6. Evaluate capital allocation (capex, dividends, acquisitions)
                7. Analyze financial health indicators
                
                Provide context for all extracted metrics and assess their significance.""",
                agent=self.create_financial_analyst_agent(),
                expected_output="""A detailed financial analysis report including:
                - Extracted financial metrics with specific figures
                - Revenue and growth trend analysis
                - Profitability and margin assessment
                - Guidance and outlook evaluation
                - Capital allocation analysis
                - Financial health summary
                - Key performance indicators identified""",
                context=[transcript_task],
                output_file=os.path.join(company_path, "financial_analysis.md")
            )
            
            # Task 3: Investment Recommendation
            recommendation_task = Task(
                description=f"""Based on the transcript and financial analysis, provide a clear 
                investment recommendation for {company_name}.
                
                Your recommendation should:
                1. Provide a clear Buy/Hold/Sell recommendation
                2. Summarize key investment thesis points (3-5 bullet points)
                3. Identify main risks and concerns
                4. Assess management quality and strategy
                5. Evaluate growth prospects and competitive position
                6. Suggest appropriate investment time horizon
                7. Provide reasoning for the recommendation
                
                Make the recommendation actionable and well-supported by the analysis.""",
                agent=self.create_investment_advisor_agent(),
                expected_output="""A final investment recommendation report including:
                - Clear recommendation (Buy/Hold/Sell) with confidence level
                - Key investment thesis (3-5 main points)
                - Primary risks and concerns to monitor
                - Management assessment
                - Growth prospects evaluation
                - Recommended time horizon
                - Executive summary for quick decision making""",
                context=[transcript_task, financial_task],
                output_file=os.path.join(company_path, "investment_recommendation.md")
            )
            
            # Create crew with simplified setup
            crew = Crew(
                agents=[
                    self.create_transcript_analyzer_agent(),
                    self.create_financial_analyst_agent(),
                    self.create_investment_advisor_agent()
                ],
                tasks=[transcript_task, financial_task, recommendation_task],
                verbose=1,  # Reduced verbosity
                process=Process.sequential
            )
            
            # Execute analysis
            print(f"Starting analysis for {company_name}...")
            result = crew.kickoff()
            
            return {
                'company': company_name,
                'status': 'success',
                'result': str(result),
                'output_files': [
                    os.path.join(company_path, "transcript_analysis.md"),
                    os.path.join(company_path, "financial_analysis.md"),
                    os.path.join(company_path, "investment_recommendation.md")
                ]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {company_name}: {e}")
            return {
                'company': company_name,
                'status': 'error',
                'message': str(e)
            }
    
    def analyze_multiple_companies(self, company_list: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple companies"""
        results = []
        
        for i, company in enumerate(company_list, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/{len(company_list)}] Analyzing {company}")
            print(f"{'='*60}")
            
            result = self.analyze_single_company(company)
            results.append(result)
            
            print(f"\nStatus: {result['status']}")
            if result['status'] == 'error':
                print(f"Error: {result['message']}")
            else:
                print("✅ Analysis completed successfully")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]]) -> str:
        """Save analysis results"""
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
            
            print(f"\n✅ Results saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return ""

def main():
    """Main execution function"""
    print("🎯 Simple Investment Analysis with CrewAI")
    print("=" * 60)
    
    # Setup configuration
    config = InvestmentAnalysisConfig()
    
    try:
        # Initialize the crew system
        analysis_crew = SimpleInvestmentAnalysisCrew(config)
        
        # Get list of companies
        if not os.path.exists(config.base_path):
            print(f"❌ Dataset not found at: {config.base_path}")
            return
        
        companies = [d for d in os.listdir(config.base_path) 
                    if os.path.isdir(os.path.join(config.base_path, d))]
        
        print(f"📊 Found {len(companies)} companies")
        
        # Interactive menu
        while True:
            print("\n" + "=" * 50)
            print("📋 ANALYSIS OPTIONS")
            print("=" * 50)
            print("1. Test with 1 company")
            print("2. Analyze 5 companies")
            print("3. Analyze custom number")
            print("4. View results")
            print("5. Exit")
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == '1':
                if companies:
                    result = analysis_crew.analyze_single_company(companies[0])
                    print(f"\n📊 Result: {result['status']}")
                    if result['status'] == 'error':
                        print(f"Error: {result['message']}")
                
            elif choice == '2':
                selected = companies[:5]
                results = analysis_crew.analyze_multiple_companies(selected)
                analysis_crew.save_results(results)
                
                success_count = sum(1 for r in results if r['status'] == 'success')
                print(f"\n📈 Summary: {success_count}/{len(results)} successful")
                
            elif choice == '3':
                try:
                    num = int(input("Enter number of companies to analyze: "))
                    selected = companies[:num]
                    results = analysis_crew.analyze_multiple_companies(selected)
                    analysis_crew.save_results(results)
                    
                    success_count = sum(1 for r in results if r['status'] == 'success')
                    print(f"\n📈 Summary: {success_count}/{len(results)} successful")
                except ValueError:
                    print("❌ Invalid number")
                    
            elif choice == '4':
                results_path = os.path.join(config.base_path, "analysis_results.csv")
                if os.path.exists(results_path):
                    df = pd.read_csv(results_path)
                    print(f"\n📊 Results Summary:")
                    print(f"Total: {len(df)}")
                    print(f"Successful: {len(df[df['status'] == 'success'])}")
                    print(f"Failed: {len(df[df['status'] == 'error'])}")
                    print("\nCompanies:")
                    for _, row in df.iterrows():
                        print(f"- {row['company']}: {row['status']}")
                else:
                    print("❌ No results found")
                    
            elif choice == '5':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()