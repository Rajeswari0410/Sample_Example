import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Model configurations
    DEFAULT_MODEL = "gpt-4-turbo-preview"
    EMBEDDING_MODEL = "text-embedding-3-small"
    
    # File paths
    TRANSCRIPTS_BASE_PATH = "MyDrive/Earnings2Insights/ECTsum"
    OUTPUT_PATH = "analysis_results"
    
    # Analysis parameters
    SENTIMENT_THRESHOLD = 0.6
    CONFIDENCE_THRESHOLD = 0.7
    MAX_TOKENS = 4000
    
    # Investment categories
    INVESTMENT_CATEGORIES = [
        "Strong Buy",
        "Buy", 
        "Hold",
        "Sell",
        "Strong Sell"
    ]
    
    # Key metrics to extract
    KEY_METRICS = [
        "revenue_growth",
        "profit_margins", 
        "cash_flow",
        "debt_levels",
        "market_outlook",
        "competitive_position",
        "management_guidance",
        "risk_factors"
    ]
    
    # Sectors for analysis
    SECTORS = [
        "Technology",
        "Healthcare", 
        "Financial Services",
        "Consumer Goods",
        "Industrial",
        "Energy",
        "Real Estate",
        "Utilities"
    ]

# Validate configuration
def validate_config():
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    return True