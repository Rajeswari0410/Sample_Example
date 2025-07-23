from crewai import Task
from typing import Dict, Any
from data_loader import TranscriptData

class TranscriptAnalysisTasks:
    """Collection of tasks for transcript analysis"""
    
    @staticmethod
    def financial_analysis_task(agent, transcript: TranscriptData) -> Task:
        """Task for financial metrics extraction and analysis"""
        return Task(
            description=f"""
            Analyze the earnings call transcript for {transcript.company_name} ({transcript.ticker}) 
            from {transcript.quarter} {transcript.year} and extract key financial metrics and insights.
            
            Focus on:
            1. Revenue growth trends and drivers
            2. Profit margins and profitability metrics
            3. Cash flow generation and management
            4. Debt levels and capital structure
            5. Return on investment metrics
            6. Guidance changes and outlook
            7. Segment performance analysis
            8. Year-over-year and quarter-over-quarter comparisons
            
            Transcript Content:
            {transcript.content[:4000]}...
            
            Provide a structured analysis with specific numbers, percentages, and clear insights.
            """,
            agent=agent,
            expected_output="""
            A comprehensive financial analysis report containing:
            - Key financial metrics summary
            - Performance trends analysis
            - Profitability assessment
            - Cash flow and balance sheet insights
            - Forward guidance evaluation
            - Risk factors identification
            """
        )
    
    @staticmethod
    def sentiment_analysis_task(agent, transcript: TranscriptData) -> Task:
        """Task for sentiment and tone analysis"""
        return Task(
            description=f"""
            Analyze the sentiment, tone, and confidence levels in the earnings call transcript 
            for {transcript.company_name} ({transcript.ticker}) from {transcript.quarter} {transcript.year}.
            
            Focus on:
            1. Overall management confidence and optimism
            2. Tone changes throughout the call
            3. Language patterns indicating stress or uncertainty
            4. Positive vs negative sentiment in different topics
            5. Management's confidence in guidance and projections
            6. Response quality and transparency in Q&A
            7. Defensive vs proactive communication style
            
            Prepared Remarks:
            {transcript.prepared_remarks[:2000]}...
            
            Q&A Section:
            {transcript.qa_section[:2000]}...
            
            Provide sentiment scores and qualitative insights.
            """,
            agent=agent,
            expected_output="""
            A detailed sentiment analysis report including:
            - Overall sentiment score (1-10 scale)
            - Management confidence assessment
            - Key tone indicators and patterns
            - Sentiment by topic areas
            - Q&A response quality evaluation
            - Red flags or concerning language patterns
            """
        )
    
    @staticmethod
    def market_analysis_task(agent, transcript: TranscriptData) -> Task:
        """Task for market and competitive analysis"""
        return Task(
            description=f"""
            Analyze market conditions, competitive positioning, and industry trends discussed 
            in the earnings call for {transcript.company_name} ({transcript.ticker}) 
            from {transcript.quarter} {transcript.year}.
            
            Focus on:
            1. Market opportunity and addressable market size
            2. Competitive positioning and market share
            3. Industry trends and dynamics
            4. Customer demand patterns and preferences
            5. Pricing power and competitive pressures
            6. New market entries or expansions
            7. Regulatory environment and impacts
            8. Technology disruptions or innovations
            
            Transcript Content:
            {transcript.content[:4000]}...
            
            Provide insights on market positioning and competitive advantages.
            """,
            agent=agent,
            expected_output="""
            A comprehensive market analysis report containing:
            - Market opportunity assessment
            - Competitive positioning analysis
            - Industry trend identification
            - Customer demand insights
            - Pricing and margin pressure analysis
            - Growth opportunity evaluation
            """
        )
    
    @staticmethod
    def risk_analysis_task(agent, transcript: TranscriptData) -> Task:
        """Task for risk assessment and identification"""
        return Task(
            description=f"""
            Identify and evaluate potential risks, challenges, and uncertainties mentioned 
            in the earnings call for {transcript.company_name} ({transcript.ticker}) 
            from {transcript.quarter} {transcript.year}.
            
            Focus on:
            1. Operational risks and challenges
            2. Financial risks and leverage concerns
            3. Market and competitive risks
            4. Regulatory and compliance risks
            5. Supply chain and operational disruptions
            6. Technology and cybersecurity risks
            7. ESG and sustainability risks
            8. Management and execution risks
            
            Transcript Content:
            {transcript.content[:4000]}...
            
            Assess both explicitly mentioned risks and implied concerns.
            """,
            agent=agent,
            expected_output="""
            A detailed risk assessment report including:
            - Risk categorization and severity levels
            - Operational risk evaluation
            - Financial risk assessment
            - Market and competitive risk analysis
            - Regulatory and compliance concerns
            - Risk mitigation strategies mentioned
            - Overall risk profile rating
            """
        )
    
    @staticmethod
    def qa_analysis_task(agent, transcript: TranscriptData) -> Task:
        """Task for Q&A session analysis"""
        return Task(
            description=f"""
            Analyze the Q&A portion of the earnings call for {transcript.company_name} 
            ({transcript.ticker}) from {transcript.quarter} {transcript.year} to extract 
            additional insights beyond the prepared remarks.
            
            Focus on:
            1. Quality and transparency of management responses
            2. Analyst concerns and focus areas
            3. Defensive or evasive responses
            4. New information revealed in Q&A
            5. Management's handling of difficult questions
            6. Consistency with prepared remarks
            7. Forward-looking statements and guidance clarifications
            
            Q&A Section:
            {transcript.qa_section}
            
            Evaluate the additional insights gained from the interactive portion.
            """,
            agent=agent,
            expected_output="""
            A Q&A analysis report containing:
            - Response quality assessment
            - Key analyst concerns identified
            - New information revealed
            - Management transparency evaluation
            - Consistency analysis with prepared remarks
            - Additional insights and red flags
            """
        )
    
    @staticmethod
    def guidance_analysis_task(agent, transcript: TranscriptData) -> Task:
        """Task for forward guidance analysis"""
        return Task(
            description=f"""
            Extract and analyze management guidance, forecasts, and forward-looking statements 
            from the earnings call for {transcript.company_name} ({transcript.ticker}) 
            from {transcript.quarter} {transcript.year}.
            
            Focus on:
            1. Specific numerical guidance provided
            2. Changes from previous guidance
            3. Confidence levels in projections
            4. Assumptions underlying guidance
            5. Scenario planning and sensitivity analysis
            6. Long-term strategic outlook
            7. Capital allocation plans
            8. Investment priorities and timing
            
            Transcript Content:
            {transcript.content[:4000]}...
            
            Evaluate the reliability and achievability of forward-looking statements.
            """,
            agent=agent,
            expected_output="""
            A forward guidance analysis report including:
            - Specific guidance metrics and ranges
            - Guidance changes and revisions
            - Underlying assumptions analysis
            - Confidence level assessment
            - Long-term outlook evaluation
            - Capital allocation insights
            - Achievability assessment
            """
        )
    
    @staticmethod
    def investment_recommendation_task(agent, analysis_results: Dict[str, Any]) -> Task:
        """Task for synthesizing analysis into investment recommendations"""
        return Task(
            description=f"""
            Synthesize the comprehensive analysis results to provide a clear investment 
            recommendation and thesis. Consider all aspects of the analysis including 
            financial metrics, sentiment, market positioning, risks, and forward guidance.
            
            Analysis Results:
            Financial Analysis: {analysis_results.get('financial_analysis', 'Not available')}
            Sentiment Analysis: {analysis_results.get('sentiment_analysis', 'Not available')}
            Market Analysis: {analysis_results.get('market_analysis', 'Not available')}
            Risk Analysis: {analysis_results.get('risk_analysis', 'Not available')}
            Q&A Analysis: {analysis_results.get('qa_analysis', 'Not available')}
            Guidance Analysis: {analysis_results.get('guidance_analysis', 'Not available')}
            
            Provide:
            1. Clear investment recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell)
            2. Investment thesis and rationale
            3. Key supporting factors
            4. Main risks and concerns
            5. Price target or valuation assessment
            6. Time horizon considerations
            7. Portfolio allocation suggestions
            """,
            agent=agent,
            expected_output="""
            A comprehensive investment recommendation report containing:
            - Clear investment rating and recommendation
            - Detailed investment thesis
            - Key bullish and bearish factors
            - Risk-reward assessment
            - Valuation analysis and price targets
            - Portfolio allocation guidance
            - Investment time horizon recommendations
            """
        )

def create_analysis_tasks(transcript: TranscriptData, agents: Dict[str, Any]) -> Dict[str, Task]:
    """Create all analysis tasks for a given transcript"""
    tasks = TranscriptAnalysisTasks()
    
    return {
        'financial_analysis': tasks.financial_analysis_task(
            agents['financial_analyst'], transcript
        ),
        'sentiment_analysis': tasks.sentiment_analysis_task(
            agents['sentiment_analyst'], transcript
        ),
        'market_analysis': tasks.market_analysis_task(
            agents['market_analyst'], transcript
        ),
        'risk_analysis': tasks.risk_analysis_task(
            agents['risk_analyst'], transcript
        ),
        'qa_analysis': tasks.qa_analysis_task(
            agents['qa_analyst'], transcript
        ),
        'guidance_analysis': tasks.guidance_analysis_task(
            agents['guidance_analyst'], transcript
        )
    }

def create_investment_recommendation_task(agents: Dict[str, Any], analysis_results: Dict[str, Any]) -> Task:
    """Create the final investment recommendation task"""
    tasks = TranscriptAnalysisTasks()
    return tasks.investment_recommendation_task(
        agents['investment_strategist'], analysis_results
    )