# Investment Analysis with CrewAI

A comprehensive AI-powered investment analysis system that analyzes company earnings transcripts and provides investment recommendations using multiple specialized AI agents.

## 🎯 Features

- **4 Specialized AI Agents:**
  - **Transcript Analyst**: Extracts key insights from earnings calls
  - **Financial Analyst**: Analyzes financial health and metrics
  - **Risk Analyst**: Identifies and assesses potential risks
  - **Investment Advisor**: Provides clear buy/hold/sell recommendations

- **Enhanced Analysis:**
  - Advanced sentiment analysis with financial context
  - Alpha Vantage integration for real-time financial data
  - Comprehensive error handling and logging
  - Structured markdown reports for each company

- **Flexible Execution:**
  - Quick test mode for single companies
  - Batch processing for multiple companies
  - Customizable analysis depth and scope

## 🚀 Quick Start (Google Colab)

### Option 1: Simple Setup Script

1. **Upload the setup script to Google Colab:**
   ```python
   # In a new Colab notebook cell, run:
   !wget https://raw.githubusercontent.com/your-repo/colab_setup.py
   exec(open('colab_setup.py').read())
   ```

2. **Or copy and paste the `colab_setup.py` content directly into a Colab cell and run it.**

### Option 2: Manual Setup

1. **Install packages:**
   ```python
   !pip install crewai crewai-tools langchain-community langchain-openai requests pandas
   ```

2. **Mount Google Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Set up API keys (optional but recommended):**
   ```python
   import os
   os.environ['OPENAI_API_KEY'] = 'your-openai-api-key'
   os.environ['ALPHA_VANTAGE_API_KEY'] = 'your-alpha-vantage-key'
   ```

4. **Run the main analysis script:**
   ```python
   exec(open('investment_analysis_crew.py').read())
   ```

## 📁 Dataset Structure

Ensure your Google Drive has the following structure:
```
/content/drive/MyDrive/Earnings2Insights/ECTsum/
├── Company1/
│   └── source/
│       └── source.md
├── Company2/
│   └── source/
│       └── source.md
└── ...
```

## 🔑 API Keys Setup

### OpenAI API Key (Recommended)
- Get your key from: https://platform.openai.com/api-keys
- Provides better performance and reliability
- Required for optimal results

### Alpha Vantage API Key (Optional)
- Get free key from: https://www.alphavantage.co/support/#api-key
- Enables real-time financial data retrieval
- Enhances financial analysis quality

## 📊 Usage Examples

### Quick Test (Single Company)
```python
# Test with one company, transcript analysis only
config = InvestmentAnalysisConfig()
crew = InvestmentAnalysisCrew(config)
result = crew.analyze_single_company("AAPL", quick_mode=True)
```

### Batch Analysis
```python
# Analyze multiple companies with full analysis
companies = ["AAPL", "MSFT", "GOOGL"]
results = crew.analyze_multiple_companies(companies, quick_mode=False)
```

### View Results
```python
# Load and display results
import pandas as pd
df = pd.read_csv("/path/to/analysis_results.csv")
print(df[['company', 'status']])
```

## 📈 Output Files

For each analyzed company, the system generates:

1. **`transcript_analysis.md`** - Detailed transcript analysis with sentiment
2. **`financial_analysis.md`** - Financial health and performance metrics
3. **`risk_assessment.md`** - Risk factors and mitigation strategies
4. **`investment_recommendation.md`** - Final buy/hold/sell recommendation

Plus summary files:
- **`analysis_results.csv`** - Complete results dataset
- **`analysis_summary.json`** - Success rates and statistics

## ⚙️ Configuration Options

### Analysis Modes
- **Quick Mode**: Transcript analysis only (faster)
- **Full Mode**: Complete 4-agent analysis (comprehensive)

### Batch Processing
- **Small Batch**: 5 companies (recommended for testing)
- **Custom Batch**: Specify number of companies
- **Full Dataset**: All 40 companies (may take hours)

### LLM Options
- **OpenAI GPT-3.5-turbo** (recommended)
- **Ollama Mistral** (local/free option)

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   Solution: Reinstall packages
   !pip install --upgrade crewai crewai-tools langchain-community
   ```

2. **Google Drive Mount Issues**
   ```python
   # Manually mount and verify
   from google.colab import drive
   drive.mount('/content/drive', force_remount=True)
   ```

3. **API Key Problems**
   ```python
   # Verify API keys are set
   import os
   print("OpenAI:", bool(os.getenv('OPENAI_API_KEY')))
   print("Alpha Vantage:", bool(os.getenv('ALPHA_VANTAGE_API_KEY')))
   ```

4. **File Path Errors**
   ```python
   # Check dataset structure
   import os
   base_path = "/content/drive/MyDrive/Earnings2Insights/ECTsum"
   if os.path.exists(base_path):
       print("✅ Dataset found")
       print("Companies:", os.listdir(base_path)[:5])
   else:
       print("❌ Dataset not found")
   ```

5. **Memory Issues**
   - Reduce batch size
   - Use quick mode
   - Restart runtime between large batches

6. **Rate Limiting**
   - Add delays between API calls
   - Use smaller batches
   - Check API quotas

### Error Codes

- **Status: 'error'** - Check error message in results
- **Status: 'success'** - Analysis completed successfully
- **Missing source.md** - Verify file structure
- **LLM setup failed** - Check API keys

## 📝 Customization

### Adding New Agents
```python
def create_custom_agent(self) -> Agent:
    return Agent(
        role="Custom Analyst",
        goal="Perform specialized analysis",
        backstory="Your agent background...",
        tools=[self.tools['search']],
        llm=self.llm
    )
```

### Custom Tools
```python
class CustomTool(BaseTool):
    name: str = "custom_tool"
    description: str = "Tool description"
    
    def _run(self, input: str) -> str:
        # Your tool logic
        return "Tool output"
```

### Analysis Customization
- Modify agent backstories for different perspectives
- Add new tools for specific data sources
- Customize task descriptions for focus areas
- Adjust output formats and file structures

## 🎯 Best Practices

1. **Start Small**: Test with 1-2 companies first
2. **Use API Keys**: OpenAI provides much better results
3. **Monitor Resources**: Watch memory and API usage
4. **Batch Processing**: Process in smaller groups for stability
5. **Error Handling**: Check results status before proceeding
6. **Backup Results**: Save important analysis outputs

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Verify your dataset structure and API keys
3. Try the quick test mode first
4. Review error messages in the results

## 🔄 Updates

- **v1.0**: Initial release with 4-agent system
- **v1.1**: Enhanced error handling and Google Colab support
- **v1.2**: Added Alpha Vantage integration and batch processing

## 📄 License

This project is open source. Please ensure you comply with API provider terms of service when using OpenAI and Alpha Vantage APIs.