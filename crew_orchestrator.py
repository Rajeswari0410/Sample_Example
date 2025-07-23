from crewai import Crew, Process
from typing import Dict, List, Any
import json
import os
from datetime import datetime
import pandas as pd

from agents import create_analysis_crew
from tasks import create_analysis_tasks, create_investment_recommendation_task
from data_loader import TranscriptData, TranscriptLoader
from config import Config, validate_config

class TranscriptAnalysisOrchestrator:
    """Main orchestrator for transcript analysis using CrewAI"""
    
    def __init__(self):
        validate_config()
        self.agents = create_analysis_crew()
        self.results = {}
        self.transcript_loader = TranscriptLoader(Config.TRANSCRIPTS_BASE_PATH)
        
        # Create output directory
        os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
    
    def analyze_single_transcript(self, transcript: TranscriptData) -> Dict[str, Any]:
        """Analyze a single transcript and return comprehensive results"""
        print(f"\n🔍 Analyzing transcript for {transcript.company_name} ({transcript.ticker}) - {transcript.quarter} {transcript.year}")
        
        # Create analysis tasks
        analysis_tasks = create_analysis_tasks(transcript, self.agents)
        
        # Create crew for parallel analysis
        analysis_crew = Crew(
            agents=list(self.agents.values())[:-1],  # Exclude investment strategist for now
            tasks=list(analysis_tasks.values()),
            process=Process.parallel,
            verbose=True,
            memory=True
        )
        
        # Execute analysis tasks
        print("📊 Running parallel analysis tasks...")
        analysis_results = analysis_crew.kickoff()
        
        # Parse results into structured format
        structured_results = self._parse_analysis_results(analysis_results, analysis_tasks)
        
        # Create investment recommendation task
        investment_task = create_investment_recommendation_task(
            self.agents, structured_results
        )
        
        # Create crew for investment recommendation
        recommendation_crew = Crew(
            agents=[self.agents['investment_strategist']],
            tasks=[investment_task],
            process=Process.sequential,
            verbose=True,
            memory=True
        )
        
        # Execute investment recommendation
        print("💡 Generating investment recommendation...")
        investment_result = recommendation_crew.kickoff()
        
        # Combine all results
        final_results = {
            'transcript_info': {
                'company_name': transcript.company_name,
                'ticker': transcript.ticker,
                'quarter': transcript.quarter,
                'year': transcript.year,
                'analysis_date': datetime.now().isoformat()
            },
            'analysis_results': structured_results,
            'investment_recommendation': investment_result,
            'metadata': {
                'content_length': len(transcript.content),
                'has_qa_section': len(transcript.qa_section) > 0,
                'file_path': transcript.file_path
            }
        }
        
        # Save results
        self._save_results(final_results, transcript)
        
        return final_results
    
    def analyze_all_transcripts(self) -> Dict[str, Any]:
        """Analyze all available transcripts"""
        print("🚀 Starting comprehensive transcript analysis...")
        
        # Load all transcripts
        transcripts = self.transcript_loader.load_all_transcripts()
        
        if not transcripts:
            print("❌ No transcripts found. Please check the data path.")
            return {}
        
        print(f"📁 Found {len(transcripts)} transcripts to analyze")
        
        all_results = {}
        summary_data = []
        
        for i, transcript in enumerate(transcripts, 1):
            print(f"\n{'='*50}")
            print(f"Processing {i}/{len(transcripts)}: {transcript.ticker}")
            print(f"{'='*50}")
            
            try:
                results = self.analyze_single_transcript(transcript)
                all_results[f"{transcript.ticker}_{transcript.quarter}_{transcript.year}"] = results
                
                # Extract summary information
                summary_data.append(self._extract_summary_info(results))
                
            except Exception as e:
                print(f"❌ Error analyzing {transcript.ticker}: {str(e)}")
                continue
        
        # Generate portfolio-level insights
        portfolio_insights = self._generate_portfolio_insights(all_results, summary_data)
        
        # Create comprehensive report
        comprehensive_report = {
            'analysis_overview': {
                'total_companies_analyzed': len(all_results),
                'analysis_date': datetime.now().isoformat(),
                'data_source': Config.TRANSCRIPTS_BASE_PATH
            },
            'individual_analyses': all_results,
            'portfolio_insights': portfolio_insights,
            'summary_table': summary_data
        }
        
        # Save comprehensive report
        self._save_comprehensive_report(comprehensive_report)
        
        return comprehensive_report
    
    def _parse_analysis_results(self, raw_results: Any, tasks: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and structure the analysis results"""
        structured_results = {}
        
        # If raw_results is a list of task results
        if isinstance(raw_results, list):
            task_names = list(tasks.keys())
            for i, result in enumerate(raw_results):
                if i < len(task_names):
                    structured_results[task_names[i]] = str(result)
        else:
            # If it's a single result or different format
            structured_results['combined_analysis'] = str(raw_results)
        
        return structured_results
    
    def _extract_summary_info(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key information for summary table"""
        transcript_info = results.get('transcript_info', {})
        investment_rec = results.get('investment_recommendation', '')
        
        # Try to extract investment rating from recommendation
        rating = self._extract_investment_rating(investment_rec)
        
        return {
            'Company': transcript_info.get('company_name', 'Unknown'),
            'Ticker': transcript_info.get('ticker', 'UNK'),
            'Quarter': transcript_info.get('quarter', 'Q1'),
            'Year': transcript_info.get('year', '2021'),
            'Investment_Rating': rating,
            'Analysis_Date': transcript_info.get('analysis_date', ''),
            'Content_Length': results.get('metadata', {}).get('content_length', 0),
            'Has_QA': results.get('metadata', {}).get('has_qa_section', False)
        }
    
    def _extract_investment_rating(self, recommendation_text: str) -> str:
        """Extract investment rating from recommendation text"""
        rating_keywords = {
            'Strong Buy': ['strong buy', 'strongly recommend'],
            'Buy': ['buy', 'recommend', 'positive'],
            'Hold': ['hold', 'neutral', 'maintain'],
            'Sell': ['sell', 'negative'],
            'Strong Sell': ['strong sell', 'avoid']
        }
        
        recommendation_lower = str(recommendation_text).lower()
        
        for rating, keywords in rating_keywords.items():
            if any(keyword in recommendation_lower for keyword in keywords):
                return rating
        
        return 'Hold'  # Default rating
    
    def _generate_portfolio_insights(self, all_results: Dict[str, Any], summary_data: List[Dict]) -> Dict[str, Any]:
        """Generate portfolio-level insights from all analyses"""
        if not summary_data:
            return {}
        
        df = pd.DataFrame(summary_data)
        
        # Rating distribution
        rating_distribution = df['Investment_Rating'].value_counts().to_dict()
        
        # Top recommendations
        top_buys = df[df['Investment_Rating'].isin(['Strong Buy', 'Buy'])]['Company'].tolist()
        
        # Sector analysis (if available)
        sectors = self._categorize_companies_by_sector(df['Company'].tolist())
        
        insights = {
            'rating_distribution': rating_distribution,
            'top_buy_recommendations': top_buys[:5],  # Top 5
            'total_companies': len(df),
            'average_content_length': df['Content_Length'].mean(),
            'companies_with_qa': df['Has_QA'].sum(),
            'sector_breakdown': sectors,
            'investment_summary': {
                'bullish_companies': len(df[df['Investment_Rating'].isin(['Strong Buy', 'Buy'])]),
                'bearish_companies': len(df[df['Investment_Rating'].isin(['Sell', 'Strong Sell'])]),
                'neutral_companies': len(df[df['Investment_Rating'] == 'Hold'])
            }
        }
        
        return insights
    
    def _categorize_companies_by_sector(self, companies: List[str]) -> Dict[str, int]:
        """Categorize companies by sector (basic implementation)"""
        # This is a simplified categorization - you could enhance with a proper mapping
        sector_keywords = {
            'Technology': ['tech', 'software', 'data', 'cloud'],
            'Financial': ['bank', 'financial', 'capital', 'investment'],
            'Healthcare': ['health', 'medical', 'pharma', 'bio'],
            'Industrial': ['industries', 'manufacturing', 'engineering'],
            'Energy': ['energy', 'oil', 'gas', 'renewable'],
            'Consumer': ['consumer', 'retail', 'goods']
        }
        
        sector_counts = {sector: 0 for sector in sector_keywords.keys()}
        sector_counts['Other'] = 0
        
        for company in companies:
            company_lower = company.lower()
            categorized = False
            
            for sector, keywords in sector_keywords.items():
                if any(keyword in company_lower for keyword in keywords):
                    sector_counts[sector] += 1
                    categorized = True
                    break
            
            if not categorized:
                sector_counts['Other'] += 1
        
        return sector_counts
    
    def _save_results(self, results: Dict[str, Any], transcript: TranscriptData):
        """Save individual analysis results"""
        filename = f"{transcript.ticker}_{transcript.quarter}_{transcript.year}_analysis.json"
        filepath = os.path.join(Config.OUTPUT_PATH, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {filepath}")
    
    def _save_comprehensive_report(self, report: Dict[str, Any]):
        """Save comprehensive analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_analysis_report_{timestamp}.json"
        filepath = os.path.join(Config.OUTPUT_PATH, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Also save summary as CSV
        if report.get('summary_table'):
            summary_df = pd.DataFrame(report['summary_table'])
            csv_filename = f"analysis_summary_{timestamp}.csv"
            csv_filepath = os.path.join(Config.OUTPUT_PATH, csv_filename)
            summary_df.to_csv(csv_filepath, index=False)
            print(f"📊 Summary CSV saved to: {csv_filepath}")
        
        print(f"📋 Comprehensive report saved to: {filepath}")
    
    def get_analysis_summary(self) -> pd.DataFrame:
        """Get summary of all completed analyses"""
        summary_files = []
        
        for filename in os.listdir(Config.OUTPUT_PATH):
            if filename.endswith('_analysis.json'):
                filepath = os.path.join(Config.OUTPUT_PATH, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        summary_files.append(self._extract_summary_info(data))
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
        
        return pd.DataFrame(summary_files) if summary_files else pd.DataFrame()

# Main execution function
def main():
    """Main function to run the transcript analysis"""
    orchestrator = TranscriptAnalysisOrchestrator()
    
    # Analyze all transcripts
    results = orchestrator.analyze_all_transcripts()
    
    if results:
        print("\n" + "="*60)
        print("🎉 ANALYSIS COMPLETE!")
        print("="*60)
        
        portfolio_insights = results.get('portfolio_insights', {})
        
        print(f"📈 Total Companies Analyzed: {portfolio_insights.get('total_companies', 0)}")
        print(f"🏆 Top Buy Recommendations: {', '.join(portfolio_insights.get('top_buy_recommendations', []))}")
        print(f"📊 Rating Distribution: {portfolio_insights.get('rating_distribution', {})}")
        print(f"🎯 Investment Summary:")
        
        inv_summary = portfolio_insights.get('investment_summary', {})
        print(f"   • Bullish: {inv_summary.get('bullish_companies', 0)} companies")
        print(f"   • Bearish: {inv_summary.get('bearish_companies', 0)} companies") 
        print(f"   • Neutral: {inv_summary.get('neutral_companies', 0)} companies")
        
        print(f"\n📁 Results saved in: {Config.OUTPUT_PATH}")
    else:
        print("❌ No results generated. Please check your data and configuration.")

if __name__ == "__main__":
    main()