from crewai import Agent
from crewai_tools import SerperDevTool, WebsiteSearchTool
from langchain_openai import ChatOpenAI
from config import Config

# Initialize the language model
llm = ChatOpenAI(
    model=Config.DEFAULT_MODEL,
    temperature=0.1,
    api_key=Config.OPENAI_API_KEY
)

class TranscriptAnalysisAgents:
    """Collection of specialized agents for transcript analysis"""
    
    @staticmethod
    def financial_analyst_agent():
        """Agent specialized in financial metrics extraction and analysis"""
        return Agent(
            role="Senior Financial Analyst",
            goal="Extract and analyze key financial metrics from earnings call transcripts to assess company performance",
            backstory="""You are a seasoned financial analyst with 15+ years of experience in equity research. 
            You specialize in analyzing earnings calls to identify key financial trends, performance indicators, 
            and potential red flags. You have a deep understanding of financial statements, ratios, and 
            industry benchmarks.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            max_iter=3,
            memory=True
        )
    
    @staticmethod
    def sentiment_analyst_agent():
        """Agent specialized in sentiment and tone analysis"""
        return Agent(
            role="Sentiment Analysis Expert",
            goal="Analyze the sentiment, tone, and confidence levels in management communications during earnings calls",
            backstory="""You are an expert in natural language processing and behavioral finance. 
            You specialize in reading between the lines of corporate communications, identifying 
            management confidence levels, market sentiment, and potential concerns that may not 
            be explicitly stated. You understand the nuances of corporate speak and can detect 
            subtle changes in tone that indicate underlying business conditions.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            max_iter=3,
            memory=True
        )
    
    @staticmethod
    def market_analyst_agent():
        """Agent specialized in market and competitive analysis"""
        return Agent(
            role="Market Research Analyst",
            goal="Analyze market conditions, competitive positioning, and industry trends mentioned in earnings calls",
            backstory="""You are a market research analyst with expertise in competitive intelligence 
            and industry analysis. You excel at identifying market opportunities, competitive threats, 
            and industry trends from corporate communications. You understand how macro-economic factors, 
            regulatory changes, and competitive dynamics impact business performance.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            max_iter=3,
            memory=True
        )
    
    @staticmethod
    def risk_analyst_agent():
        """Agent specialized in risk assessment and identification"""
        return Agent(
            role="Risk Assessment Specialist",
            goal="Identify and evaluate potential risks, challenges, and uncertainties mentioned in earnings calls",
            backstory="""You are a risk management expert with deep experience in identifying 
            and quantifying business risks. You specialize in parsing corporate communications 
            to identify operational, financial, regulatory, and strategic risks that could impact 
            future performance. You understand how to translate qualitative risk discussions 
            into actionable risk assessments.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            max_iter=3,
            memory=True
        )
    
    @staticmethod
    def investment_strategist_agent():
        """Agent that synthesizes all analysis into investment recommendations"""
        return Agent(
            role="Senior Investment Strategist",
            goal="Synthesize financial, sentiment, market, and risk analysis to provide comprehensive investment recommendations",
            backstory="""You are a senior investment strategist with 20+ years of experience in 
            portfolio management and equity research. You excel at synthesizing complex financial 
            and qualitative information to make clear, actionable investment recommendations. 
            You understand how to balance potential returns with risks and can communicate 
            investment thesis clearly to both institutional and retail investors.""",
            verbose=True,
            allow_delegation=True,
            llm=llm,
            max_iter=5,
            memory=True
        )
    
    @staticmethod
    def qa_analyst_agent():
        """Agent specialized in analyzing Q&A sessions for additional insights"""
        return Agent(
            role="Q&A Session Analyst",
            goal="Analyze Q&A portions of earnings calls to extract additional insights about management responses and analyst concerns",
            backstory="""You are an expert in analyzing Q&A sessions from earnings calls. 
            You understand that the Q&A portion often reveals more candid insights about 
            business challenges, management priorities, and market concerns. You excel at 
            identifying patterns in analyst questions and evaluating the quality and 
            transparency of management responses.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            max_iter=3,
            memory=True
        )
    
    @staticmethod
    def guidance_analyst_agent():
        """Agent specialized in analyzing forward-looking guidance and projections"""
        return Agent(
            role="Forward Guidance Analyst",
            goal="Extract and analyze management guidance, forecasts, and forward-looking statements",
            backstory="""You are a specialist in analyzing corporate guidance and forward-looking 
            statements. You understand how to interpret management projections, identify changes 
            in guidance patterns, and assess the reliability of forward-looking statements based 
            on historical accuracy and current business conditions. You excel at translating 
            qualitative guidance into quantitative expectations.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            max_iter=3,
            memory=True
        )

class AgentTools:
    """Tools that can be used by agents for enhanced analysis"""
    
    @staticmethod
    def get_web_search_tool():
        """Web search tool for additional context"""
        return SerperDevTool()
    
    @staticmethod
    def get_website_search_tool():
        """Website search tool for company-specific information"""
        return WebsiteSearchTool()

# Agent factory function
def create_analysis_crew():
    """Create a crew of agents for comprehensive transcript analysis"""
    agents = TranscriptAnalysisAgents()
    
    return {
        'financial_analyst': agents.financial_analyst_agent(),
        'sentiment_analyst': agents.sentiment_analyst_agent(),
        'market_analyst': agents.market_analyst_agent(),
        'risk_analyst': agents.risk_analyst_agent(),
        'investment_strategist': agents.investment_strategist_agent(),
        'qa_analyst': agents.qa_analyst_agent(),
        'guidance_analyst': agents.guidance_analyst_agent()
    }