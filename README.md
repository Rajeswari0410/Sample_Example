# 📊 Earnings Call Investment Insights System

A comprehensive AI-powered system that analyzes earnings call transcripts to provide intelligent investment recommendations using CrewAI agents.

## 🚀 Features

### 🤖 Multi-Agent Analysis
- **Financial Analyst Agent**: Extracts key financial metrics and performance indicators
- **Sentiment Analyst Agent**: Analyzes management tone, confidence, and communication patterns
- **Market Analyst Agent**: Evaluates competitive positioning and industry trends
- **Risk Analyst Agent**: Identifies potential risks and challenges
- **Q&A Analyst Agent**: Analyzes question-and-answer sessions for additional insights
- **Guidance Analyst Agent**: Evaluates forward-looking statements and projections
- **Investment Strategist Agent**: Synthesizes all analyses into actionable recommendations

### 📈 Investment Intelligence
- **Automated Investment Ratings**: Strong Buy, Buy, Hold, Sell, Strong Sell
- **Comprehensive Analysis**: Financial metrics, sentiment, market position, and risks
- **Portfolio-Level Insights**: Cross-company comparisons and trend analysis
- **Interactive Dashboard**: Streamlit-based visualization and exploration

### 🔍 Advanced Analytics
- **Sentiment Analysis**: Management confidence and tone assessment
- **Risk Assessment**: Operational, financial, and strategic risk identification
- **Trend Analysis**: Performance patterns across quarters and years
- **Comparative Analysis**: Cross-company benchmarking

## 🏗️ System Architecture

```
📁 Earnings Call Investment System
├── 🤖 CrewAI Agents (agents.py)
├── 📋 Task Definitions (tasks.py)
├── 🎯 Orchestration Engine (crew_orchestrator.py)
├── 📊 Data Processing (data_loader.py)
├── 🖥️ Interactive Dashboard (streamlit_dashboard.py)
├── ⚙️ Configuration (config.py)
└── 📈 Results & Reports (analysis_results/)
```

## 🛠️ Installation & Setup

### 1. Clone and Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys
nano .env
```

Required environment variables:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Data Setup

Place your earnings call transcript files in the following structure:
```
MyDrive/Earnings2Insights/ECTsum/
├── ABM_q3_2021/
│   └── source/
│       └── source.md
├── AME_q1_2021/
│   └── source/
│       └── source.md
└── CFR_q3_2019/
    └── source/
        └── source.md
```

## 🚀 Usage

### Command Line Analysis

```bash
# Run comprehensive analysis on all transcripts
python crew_orchestrator.py
```

### Interactive Dashboard

```bash
# Launch the Streamlit dashboard
streamlit run streamlit_dashboard.py
```

### Programmatic Usage

```python
from crew_orchestrator import TranscriptAnalysisOrchestrator

# Initialize the orchestrator
orchestrator = TranscriptAnalysisOrchestrator()

# Analyze all transcripts
results = orchestrator.analyze_all_transcripts()

# Get summary
summary = orchestrator.get_analysis_summary()
print(summary)
```

## 📊 Sample Analysis Output

### Investment Recommendation Example
```json
{
  "transcript_info": {
    "company_name": "ABM Industries",
    "ticker": "ABM",
    "quarter": "Q3",
    "year": "2021",
    "analysis_date": "2024-01-15T10:30:00"
  },
  "analysis_results": {
    "financial_analysis": "Strong revenue growth of 10.7%...",
    "sentiment_analysis": "Management confidence score: 8/10...",
    "market_analysis": "Competitive positioning improved...",
    "risk_analysis": "Labor market challenges identified...",
    "qa_analysis": "Transparent responses to analyst questions...",
    "guidance_analysis": "Raised full-year EPS guidance..."
  },
  "investment_recommendation": "BUY - Strong operational performance and positive outlook justify a buy recommendation..."
}
```

### Portfolio Summary
```
📈 Total Companies Analyzed: 40
🏆 Top Buy Recommendations: ABM Industries, AMETEK Inc, Cullen/Frost Bankers
📊 Rating Distribution: {'Buy': 15, 'Hold': 20, 'Sell': 5}
🎯 Investment Summary:
   • Bullish: 15 companies
   • Bearish: 5 companies
   • Neutral: 20 companies
```

## 🎛️ Dashboard Features

### 📈 Portfolio Overview
- Investment rating distribution
- Performance trends over time
- Top buy/sell recommendations
- Sector analysis

### 🏢 Company Analysis
- Individual company deep-dives
- Detailed investment recommendations
- Multi-dimensional analysis results
- Historical performance tracking

### 📊 Comparative Analysis
- Cross-company benchmarking
- Content length vs. rating analysis
- Q&A session insights
- Temporal trend analysis

### 🔍 Detailed Insights
- Key trend identification
- Risk pattern analysis
- Sentiment evolution
- Market opportunity assessment

## 🔧 Configuration

### Model Settings
```python
# config.py
DEFAULT_MODEL = "gpt-4-turbo-preview"
EMBEDDING_MODEL = "text-embedding-3-small"
SENTIMENT_THRESHOLD = 0.6
CONFIDENCE_THRESHOLD = 0.7
```

### Analysis Parameters
```python
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
```

## 📁 Output Structure

```
analysis_results/
├── ABM_Q3_2021_analysis.json
├── AME_Q1_2021_analysis.json
├── comprehensive_analysis_report_20240115_103000.json
└── analysis_summary_20240115_103000.csv
```

## 🎯 Use Cases

### 📈 Investment Research
- **Equity Research**: Comprehensive company analysis for investment decisions
- **Portfolio Management**: Multi-company comparison and ranking
- **Risk Assessment**: Systematic risk identification across holdings

### 🏢 Corporate Analysis
- **Competitive Intelligence**: Market positioning and competitive analysis
- **Management Assessment**: Leadership confidence and communication evaluation
- **Strategic Planning**: Market trend and opportunity identification

### 📊 Financial Analysis
- **Performance Tracking**: Quarter-over-quarter and year-over-year analysis
- **Guidance Evaluation**: Forward-looking statement reliability assessment
- **Sentiment Monitoring**: Management tone and confidence tracking

## 🔮 Advanced Features

### 🤖 Agent Customization
```python
# Custom agent creation
def create_custom_analyst():
    return Agent(
        role="ESG Analyst",
        goal="Analyze ESG factors mentioned in earnings calls",
        backstory="Expert in environmental, social, and governance analysis...",
        llm=llm
    )
```

### 📊 Custom Metrics
```python
# Add custom analysis metrics
CUSTOM_METRICS = [
    "esg_factors",
    "digital_transformation",
    "supply_chain_resilience",
    "customer_satisfaction"
]
```

### 🎨 Dashboard Customization
- Custom visualizations
- Additional filter options
- Export capabilities
- Real-time updates

## 🔍 Troubleshooting

### Common Issues

1. **API Key Errors**
   ```bash
   # Ensure your .env file has the correct API key
   OPENAI_API_KEY=sk-...
   ```

2. **Data Path Issues**
   ```python
   # Update the base path in config.py
   TRANSCRIPTS_BASE_PATH = "your/custom/path"
   ```

3. **Memory Issues**
   ```python
   # Reduce max_tokens in config.py
   MAX_TOKENS = 2000
   ```

### Performance Optimization

- **Parallel Processing**: Agents run in parallel for faster analysis
- **Caching**: Results are cached to avoid re-processing
- **Streaming**: Large datasets processed in chunks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CrewAI**: Multi-agent orchestration framework
- **OpenAI**: GPT models for natural language processing
- **Streamlit**: Interactive dashboard framework
- **Plotly**: Advanced data visualization

## 📞 Support

For questions, issues, or feature requests:
- 📧 Email: support@earningscall-insights.com
- 💬 GitHub Issues: [Create an issue](https://github.com/your-repo/issues)
- 📖 Documentation: [Full Documentation](https://docs.earningscall-insights.com)

---

**Made with ❤️ for intelligent investment analysis**