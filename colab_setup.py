#!/usr/bin/env python3
"""
Google Colab Setup Script for Investment Analysis with CrewAI
Run this script in Google Colab to set up and execute the investment analysis system.
"""

# ============================================================================
# STEP 1: INSTALL REQUIRED PACKAGES
# ============================================================================

def install_packages():
    """Install all required packages for the investment analysis system"""
    import subprocess
    import sys
    
    packages = [
        "crewai",
        "crewai-tools", 
        "langchain-community",
        "langchain-openai",
        "requests",
        "pandas",
        "python-dotenv"
    ]
    
    print("📦 Installing required packages...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
    
    print("\n🎉 Package installation complete!")

# ============================================================================
# STEP 2: MOUNT GOOGLE DRIVE
# ============================================================================

def mount_drive_and_verify():
    """Mount Google Drive and verify dataset location"""
    try:
        from google.colab import drive
        import os
        
        # Mount Google Drive
        print("📁 Mounting Google Drive...")
        drive.mount('/content/drive')
        
        # Verify the dataset path
        base_path = "/content/drive/MyDrive/Earnings2Insights/ECTsum"
        if os.path.exists(base_path):
            print(f"✅ Found dataset at: {base_path}")
            
            # List companies
            companies = [d for d in os.listdir(base_path) 
                        if os.path.isdir(os.path.join(base_path, d))]
            print(f"📊 Found {len(companies)} companies")
            
            # Show first few companies
            if companies:
                print(f"📋 Sample companies: {companies[:5]}{'...' if len(companies) > 5 else ''}")
            
            return base_path, companies
        else:
            print(f"❌ Dataset not found at: {base_path}")
            print("Please check your Google Drive path and folder structure.")
            return None, []
            
    except ImportError:
        print("❌ This script is designed for Google Colab")
        return None, []
    except Exception as e:
        print(f"❌ Error mounting drive: {e}")
        return None, []

# ============================================================================
# STEP 3: CONFIGURE API KEYS
# ============================================================================

def setup_api_keys():
    """Setup API keys for enhanced functionality"""
    import os
    from getpass import getpass
    
    print("🔑 API Key Configuration")
    print("=" * 50)
    
    # OpenAI API Key (recommended for better performance)
    print("For best results, provide an OpenAI API key:")
    openai_key = getpass("Enter OpenAI API Key (or press Enter to skip): ")
    if openai_key.strip():
        os.environ['OPENAI_API_KEY'] = openai_key.strip()
        print("✅ OpenAI API key set")
    else:
        print("⚠️ No OpenAI key provided. Will attempt to use Ollama (may require setup)")
    
    # Alpha Vantage API Key (optional, for financial data)
    print("\nFor enhanced financial data, provide an Alpha Vantage API key:")
    print("(Get free key at: https://www.alphavantage.co/support/#api-key)")
    alpha_key = getpass("Enter Alpha Vantage API Key (or press Enter to skip): ")
    if alpha_key.strip():
        os.environ['ALPHA_VANTAGE_API_KEY'] = alpha_key.strip()
        print("✅ Alpha Vantage API key set")
    else:
        print("⚠️ No Alpha Vantage key provided. Will use basic financial data")
    
    print("\n🚀 Configuration complete!")
    return bool(openai_key.strip()), bool(alpha_key.strip())

# ============================================================================
# STEP 4: LOAD INVESTMENT ANALYSIS SYSTEM
# ============================================================================

def create_analysis_system():
    """Create the investment analysis system"""
    
    # Import the investment analysis code
    analysis_code = '''
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
    raise

class InvestmentAnalysisConfig:
    """Configuration class for the investment analysis system"""
    
    def __init__(self):
        self.base_path = "/content/drive/MyDrive/Earnings2Insights/ECTsum"
        self.use_openai = bool(os.getenv('OPENAI_API_KEY'))
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
                if self.openai_api_key:
                    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
                else:
                    raise ValueError("No valid LLM configuration found")

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
            positive_words = [
                'growth', 'profit', 'increase', 'strong', 'success', 'revenue',
                'expansion', 'bullish', 'outperform', 'beat', 'exceed', 'positive',
                'improvement', 'momentum', 'opportunity', 'optimistic'
            ]
            
            negative_words = [
                'loss', 'decline', 'risk', 'weak', 'challenge', 'decrease',
                'bearish', 'underperform', 'miss', 'concern', 'negative',
                'uncertainty', 'volatility', 'pressure', 'headwind'
            ]
            
            text_lower = text.lower()
            positive_count = sum(text_lower.count(word) for word in positive_words)
            negative_count = sum(text_lower.count(word) for word in negative_words)
            
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

class InvestmentAnalysisCrew:
    """Main class for running investment analysis using CrewAI"""
    
    def __init__(self, config: InvestmentAnalysisConfig):
        self.config = config
        self.llm = config.setup_llm()
        self.tools = self._setup_tools()
        
    def _setup_tools(self):
        """Setup all tools for the crew"""
        return {
            'read_md': ReadMarkdownFileTool(),
            'sentiment': EnhancedSentimentTool(),
            'search': DuckDuckGoSearchRun()
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
    
    def analyze_single_company(self, company_name: str, quick_mode: bool = False) -> Dict[str, Any]:
        """Analyze a single company"""
        try:
            company_path = os.path.join(self.config.base_path, company_name)
            source_md_path = os.path.join(company_path, "source", "source.md")
            
            if not os.path.exists(source_md_path):
                return {
                    'company': company_name,
                    'status': 'error',
                    'message': f'Source file not found: {source_md_path}'
                }
            
            os.makedirs(company_path, exist_ok=True)
            
            # Create tasks
            tasks = []
            
            # Transcript Analysis Task
            transcript_task = Task(
                description=f"""Analyze the earnings transcript for {company_name} located at {source_md_path}.
                
                Extract and analyze:
                1. Key business highlights and achievements
                2. Management commentary on performance
                3. Growth strategies and initiatives
                4. Challenges and concerns mentioned
                5. Forward-looking statements and guidance
                6. Sentiment analysis of management tone
                
                Provide a comprehensive summary with specific quotes and data points.""",
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
            tasks.append(transcript_task)
            
            if not quick_mode:
                # Financial Analysis Task
                financial_task = Task(
                    description=f"""Conduct financial analysis of {company_name} based on the transcript analysis.
                    
                    Focus on:
                    1. Revenue trends and growth drivers
                    2. Profitability and margin analysis
                    3. Key financial metrics
                    4. Competitive positioning
                    5. Growth prospects
                    
                    Use available tools to gather additional market information.""",
                    agent=self.create_financial_analyst_agent(),
                    expected_output="""A financial analysis report including:
                    - Revenue and profitability assessment
                    - Key financial metrics
                    - Competitive position
                    - Growth prospects evaluation""",
                    context=[transcript_task],
                    output_file=os.path.join(company_path, "financial_analysis.md")
                )
                tasks.append(financial_task)
                
                # Investment Recommendation Task
                recommendation_task = Task(
                    description=f"""Provide investment recommendation for {company_name} based on all analysis.
                    
                    Your recommendation should:
                    1. Provide clear Buy/Hold/Sell recommendation
                    2. Include reasoning based on analysis
                    3. List 3-5 key supporting arguments
                    4. Identify main risks
                    5. Suggest investment time horizon""",
                    agent=self.create_investment_advisor_agent(),
                    expected_output="""Investment recommendation report including:
                    - Clear recommendation (Buy/Hold/Sell)
                    - Key investment thesis points
                    - Main risks to consider
                    - Time horizon and rationale""",
                    context=[transcript_task, financial_task],
                    output_file=os.path.join(company_path, "investment_recommendation.md")
                )
                tasks.append(recommendation_task)
            
            # Create crew
            agents = [self.create_transcript_analyzer_agent()]
            if not quick_mode:
                agents.extend([
                    self.create_financial_analyst_agent(),
                    self.create_investment_advisor_agent()
                ])
            
            crew = Crew(
                agents=agents,
                tasks=tasks,
                verbose=2,
                process=Process.sequential
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
    
    def analyze_multiple_companies(self, company_list: List[str], quick_mode: bool = False) -> List[Dict[str, Any]]:
        """Analyze multiple companies"""
        results = []
        
        for i, company in enumerate(company_list, 1):
            print(f"\\n[{i}/{len(company_list)}] Analyzing {company}...")
            result = self.analyze_single_company(company, quick_mode)
            results.append(result)
            
            print(f"Status: {result['status']}")
            if result['status'] == 'error':
                print(f"Error: {result['message']}")
            print("=" * 60)
        
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
            
            print(f"✅ Results saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return ""
'''
    
    # Execute the code to make classes available
    exec(analysis_code, globals())
    print("✅ Investment Analysis System loaded successfully!")

# ============================================================================
# STEP 5: MAIN EXECUTION FUNCTIONS
# ============================================================================

def run_quick_test(companies: List[str]):
    """Run a quick test on one company"""
    if not companies:
        print("❌ No companies available for testing")
        return None
    
    print("🧪 Running Quick Test...")
    print("=" * 50)
    
    config = InvestmentAnalysisConfig()
    analysis_crew = InvestmentAnalysisCrew(config)
    
    test_company = companies[0]
    print(f"Testing with: {test_company}")
    
    result = analysis_crew.analyze_single_company(test_company, quick_mode=True)
    
    print(f"\n📊 Test Result:")
    print(f"Company: {result['company']}")
    print(f"Status: {result['status']}")
    
    if result['status'] == 'success':
        print("✅ Test successful!")
        print(f"Output files: {result.get('output_files', [])}")
    else:
        print(f"❌ Test failed: {result.get('message', 'Unknown error')}")
    
    return result

def run_batch_analysis(companies: List[str], batch_size: int = 5, quick_mode: bool = False):
    """Run analysis on a batch of companies"""
    print(f"🚀 Running Batch Analysis...")
    print(f"Companies: {batch_size}, Quick Mode: {quick_mode}")
    print("=" * 50)
    
    config = InvestmentAnalysisConfig()
    analysis_crew = InvestmentAnalysisCrew(config)
    
    # Select companies for analysis
    selected_companies = companies[:batch_size]
    print(f"Selected companies: {selected_companies}")
    
    # Run analysis
    results = analysis_crew.analyze_multiple_companies(selected_companies, quick_mode)
    
    # Save results
    output_path = analysis_crew.save_results(results)
    
    # Print summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    total_count = len(results)
    
    print("\n" + "=" * 60)
    print("📈 ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Total companies: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")
    print(f"Success rate: {(success_count/total_count)*100:.1f}%" if total_count > 0 else "0%")
    
    if output_path:
        print(f"Results saved to: {output_path}")
    
    return results

def view_results(base_path: str):
    """View analysis results"""
    results_path = os.path.join(base_path, "analysis_results.csv")
    
    if os.path.exists(results_path):
        import pandas as pd
        df = pd.read_csv(results_path)
        
        print("📊 Analysis Results Summary:")
        print(f"Total companies: {len(df)}")
        print(f"Successful: {len(df[df['status'] == 'success'])}")
        print(f"Failed: {len(df[df['status'] == 'error'])}")
        
        print("\n📈 Results:")
        print(df[['company', 'status']].to_string(index=False))
        
        return df
    else:
        print("❌ No results file found")
        return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("🎯 Investment Analysis with CrewAI - Google Colab Setup")
    print("=" * 70)
    
    # Step 1: Install packages
    install_packages()
    
    # Step 2: Mount drive and verify dataset
    base_path, companies = mount_drive_and_verify()
    if not base_path:
        print("❌ Setup failed. Please check your Google Drive setup.")
        return
    
    # Step 3: Setup API keys
    has_openai, has_alpha = setup_api_keys()
    
    # Step 4: Load analysis system
    create_analysis_system()
    
    # Step 5: Interactive menu
    while True:
        print("\n" + "=" * 50)
        print("📋 ANALYSIS OPTIONS")
        print("=" * 50)
        print("1. Run Quick Test (1 company, transcript only)")
        print("2. Run Small Batch (5 companies, full analysis)")
        print("3. Run Custom Batch")
        print("4. View Results")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            run_quick_test(companies)
        
        elif choice == '2':
            run_batch_analysis(companies, batch_size=5, quick_mode=False)
        
        elif choice == '3':
            try:
                batch_size = int(input("Enter number of companies to analyze: "))
                quick_mode = input("Quick mode? (y/n): ").lower() == 'y'
                run_batch_analysis(companies, batch_size=batch_size, quick_mode=quick_mode)
            except ValueError:
                print("❌ Invalid input")
        
        elif choice == '4':
            view_results(base_path)
        
        elif choice == '5':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()