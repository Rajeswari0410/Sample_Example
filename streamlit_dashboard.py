import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
from typing import Dict, Any, List
import re

from crew_orchestrator import TranscriptAnalysisOrchestrator
from data_loader import TranscriptLoader
from config import Config

# Page configuration
st.set_page_config(
    page_title="📊 Earnings Call Investment Insights",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class InvestmentDashboard:
    """Interactive dashboard for transcript analysis results"""
    
    def __init__(self):
        self.orchestrator = None
        self.results_data = {}
        self.summary_df = pd.DataFrame()
        
    def load_data(self):
        """Load analysis results from saved files"""
        if not os.path.exists(Config.OUTPUT_PATH):
            st.error(f"Results directory not found: {Config.OUTPUT_PATH}")
            return False
        
        # Load individual analysis files
        analysis_files = [f for f in os.listdir(Config.OUTPUT_PATH) if f.endswith('_analysis.json')]
        
        if not analysis_files:
            st.warning("No analysis results found. Please run the analysis first.")
            return False
        
        summary_data = []
        
        for filename in analysis_files:
            filepath = os.path.join(Config.OUTPUT_PATH, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    key = filename.replace('_analysis.json', '')
                    self.results_data[key] = data
                    
                    # Extract summary info
                    summary_info = self._extract_summary_from_result(data)
                    summary_data.append(summary_info)
                    
            except Exception as e:
                st.error(f"Error loading {filename}: {e}")
        
        self.summary_df = pd.DataFrame(summary_data)
        return True
    
    def _extract_summary_from_result(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract summary information from result data"""
        transcript_info = result_data.get('transcript_info', {})
        investment_rec = str(result_data.get('investment_recommendation', ''))
        
        # Extract investment rating
        rating = self._extract_investment_rating(investment_rec)
        
        # Extract key metrics from analysis
        analysis_results = result_data.get('analysis_results', {})
        
        return {
            'Company': transcript_info.get('company_name', 'Unknown'),
            'Ticker': transcript_info.get('ticker', 'UNK'),
            'Quarter': transcript_info.get('quarter', 'Q1'),
            'Year': transcript_info.get('year', '2021'),
            'Investment_Rating': rating,
            'Analysis_Date': transcript_info.get('analysis_date', ''),
            'Content_Length': result_data.get('metadata', {}).get('content_length', 0),
            'Has_QA': result_data.get('metadata', {}).get('has_qa_section', False),
            'File_Path': result_data.get('metadata', {}).get('file_path', '')
        }
    
    def _extract_investment_rating(self, recommendation_text: str) -> str:
        """Extract investment rating from recommendation text"""
        rating_keywords = {
            'Strong Buy': ['strong buy', 'strongly recommend', 'buy rating'],
            'Buy': ['buy', 'recommend', 'positive outlook', 'bullish'],
            'Hold': ['hold', 'neutral', 'maintain', 'fair value'],
            'Sell': ['sell', 'negative', 'bearish', 'underperform'],
            'Strong Sell': ['strong sell', 'avoid', 'significant concerns']
        }
        
        recommendation_lower = recommendation_text.lower()
        
        for rating, keywords in rating_keywords.items():
            if any(keyword in recommendation_lower for keyword in keywords):
                return rating
        
        return 'Hold'  # Default rating
    
    def render_dashboard(self):
        """Render the main dashboard"""
        st.title("📊 Earnings Call Investment Insights Dashboard")
        st.markdown("---")
        
        # Load data
        if not self.load_data():
            st.stop()
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        if len(self.summary_df) == 0:
            st.error("No data available to display")
            return
        
        # Overview metrics
        self.render_overview_metrics()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Portfolio Overview", 
            "🏢 Company Analysis", 
            "📊 Comparative Analysis",
            "🔍 Detailed Insights",
            "📋 Raw Data"
        ])
        
        with tab1:
            self.render_portfolio_overview()
        
        with tab2:
            self.render_company_analysis()
        
        with tab3:
            self.render_comparative_analysis()
        
        with tab4:
            self.render_detailed_insights()
        
        with tab5:
            self.render_raw_data()
    
    def render_sidebar(self):
        """Render the sidebar with filters and controls"""
        st.sidebar.header("🎛️ Dashboard Controls")
        
        # Data refresh
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()
        
        # Filters
        st.sidebar.subheader("📊 Filters")
        
        # Investment rating filter
        ratings = ['All'] + list(self.summary_df['Investment_Rating'].unique())
        selected_rating = st.sidebar.selectbox("Investment Rating", ratings)
        
        # Company filter
        companies = ['All'] + list(self.summary_df['Company'].unique())
        selected_company = st.sidebar.selectbox("Company", companies)
        
        # Year filter
        years = ['All'] + sorted(list(self.summary_df['Year'].unique()))
        selected_year = st.sidebar.selectbox("Year", years)
        
        # Apply filters
        filtered_df = self.summary_df.copy()
        
        if selected_rating != 'All':
            filtered_df = filtered_df[filtered_df['Investment_Rating'] == selected_rating]
        
        if selected_company != 'All':
            filtered_df = filtered_df[filtered_df['Company'] == selected_company]
        
        if selected_year != 'All':
            filtered_df = filtered_df[filtered_df['Year'] == selected_year]
        
        self.filtered_df = filtered_df
        
        # Display filter results
        st.sidebar.markdown(f"**Showing {len(filtered_df)} of {len(self.summary_df)} companies**")
        
        # Analysis controls
        st.sidebar.subheader("⚙️ Analysis Controls")
        
        if st.sidebar.button("🚀 Run New Analysis"):
            self.run_new_analysis()
    
    def render_overview_metrics(self):
        """Render key overview metrics"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Companies", len(self.summary_df))
        
        with col2:
            buy_count = len(self.summary_df[self.summary_df['Investment_Rating'].isin(['Strong Buy', 'Buy'])])
            st.metric("Buy Recommendations", buy_count)
        
        with col3:
            hold_count = len(self.summary_df[self.summary_df['Investment_Rating'] == 'Hold'])
            st.metric("Hold Recommendations", hold_count)
        
        with col4:
            sell_count = len(self.summary_df[self.summary_df['Investment_Rating'].isin(['Sell', 'Strong Sell'])])
            st.metric("Sell Recommendations", sell_count)
        
        with col5:
            avg_content = int(self.summary_df['Content_Length'].mean()) if len(self.summary_df) > 0 else 0
            st.metric("Avg Content Length", f"{avg_content:,}")
    
    def render_portfolio_overview(self):
        """Render portfolio overview visualizations"""
        st.subheader("📈 Investment Portfolio Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Investment rating distribution
            rating_counts = self.filtered_df['Investment_Rating'].value_counts()
            
            fig_pie = px.pie(
                values=rating_counts.values,
                names=rating_counts.index,
                title="Investment Rating Distribution",
                color_discrete_map={
                    'Strong Buy': '#00CC96',
                    'Buy': '#19D3F3',
                    'Hold': '#FFA500',
                    'Sell': '#FF6692',
                    'Strong Sell': '#EF553B'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Companies by year
            year_counts = self.filtered_df['Year'].value_counts().sort_index()
            
            fig_bar = px.bar(
                x=year_counts.index,
                y=year_counts.values,
                title="Companies Analyzed by Year",
                labels={'x': 'Year', 'y': 'Number of Companies'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Investment recommendations by company
        st.subheader("🏆 Top Investment Recommendations")
        
        buy_companies = self.filtered_df[
            self.filtered_df['Investment_Rating'].isin(['Strong Buy', 'Buy'])
        ].sort_values('Investment_Rating')
        
        if len(buy_companies) > 0:
            st.dataframe(
                buy_companies[['Company', 'Ticker', 'Investment_Rating', 'Quarter', 'Year']],
                use_container_width=True
            )
        else:
            st.info("No buy recommendations in the filtered data.")
    
    def render_company_analysis(self):
        """Render individual company analysis"""
        st.subheader("🏢 Individual Company Analysis")
        
        # Company selector
        selected_company = st.selectbox(
            "Select a company for detailed analysis:",
            self.filtered_df['Company'].unique()
        )
        
        if selected_company:
            company_data = self.filtered_df[self.filtered_df['Company'] == selected_company].iloc[0]
            
            # Company info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Company", company_data['Company'])
            with col2:
                st.metric("Ticker", company_data['Ticker'])
            with col3:
                st.metric("Investment Rating", company_data['Investment_Rating'])
            with col4:
                st.metric("Quarter", f"{company_data['Quarter']} {company_data['Year']}")
            
            # Get detailed results
            company_key = f"{company_data['Ticker']}_{company_data['Quarter']}_{company_data['Year']}"
            
            if company_key in self.results_data:
                detailed_results = self.results_data[company_key]
                
                # Investment recommendation
                st.subheader("💡 Investment Recommendation")
                investment_rec = detailed_results.get('investment_recommendation', 'Not available')
                st.write(investment_rec)
                
                # Analysis results
                st.subheader("📊 Detailed Analysis")
                
                analysis_results = detailed_results.get('analysis_results', {})
                
                for analysis_type, result in analysis_results.items():
                    with st.expander(f"📋 {analysis_type.replace('_', ' ').title()}"):
                        st.write(result)
    
    def render_comparative_analysis(self):
        """Render comparative analysis across companies"""
        st.subheader("📊 Comparative Analysis")
        
        # Content length vs investment rating
        fig_scatter = px.scatter(
            self.filtered_df,
            x='Content_Length',
            y='Investment_Rating',
            color='Investment_Rating',
            size='Content_Length',
            hover_data=['Company', 'Ticker'],
            title="Content Length vs Investment Rating"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Rating distribution by quarter
        if len(self.filtered_df) > 0:
            rating_quarter = pd.crosstab(
                self.filtered_df['Quarter'], 
                self.filtered_df['Investment_Rating']
            )
            
            fig_heatmap = px.imshow(
                rating_quarter.values,
                labels=dict(x="Investment Rating", y="Quarter", color="Count"),
                x=rating_quarter.columns,
                y=rating_quarter.index,
                title="Investment Ratings by Quarter"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Companies with Q&A sessions
        qa_analysis = self.filtered_df.groupby('Has_QA').size()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Companies with Q&A", qa_analysis.get(True, 0))
        with col2:
            st.metric("Companies without Q&A", qa_analysis.get(False, 0))
    
    def render_detailed_insights(self):
        """Render detailed insights and trends"""
        st.subheader("🔍 Detailed Insights & Trends")
        
        # Key insights
        insights = self._generate_insights()
        
        for insight_type, insight_data in insights.items():
            with st.expander(f"📈 {insight_type}"):
                st.write(insight_data)
        
        # Trend analysis
        if len(self.filtered_df) > 1:
            st.subheader("📈 Trend Analysis")
            
            # Rating trends over time
            trend_data = self.filtered_df.groupby(['Year', 'Investment_Rating']).size().unstack(fill_value=0)
            
            fig_trend = go.Figure()
            for rating in trend_data.columns:
                fig_trend.add_trace(go.Scatter(
                    x=trend_data.index,
                    y=trend_data[rating],
                    mode='lines+markers',
                    name=rating,
                    line=dict(width=3)
                ))
            
            fig_trend.update_layout(
                title="Investment Rating Trends Over Time",
                xaxis_title="Year",
                yaxis_title="Number of Companies",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
    
    def render_raw_data(self):
        """Render raw data tables"""
        st.subheader("📋 Raw Analysis Data")
        
        # Summary table
        st.subheader("📊 Summary Table")
        st.dataframe(self.filtered_df, use_container_width=True)
        
        # Download options
        csv = self.filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Summary as CSV",
            data=csv,
            file_name=f"investment_analysis_summary_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Detailed results
        st.subheader("🔍 Detailed Results")
        
        selected_detail_company = st.selectbox(
            "Select company for detailed JSON view:",
            self.filtered_df['Company'].unique(),
            key="detail_selector"
        )
        
        if selected_detail_company:
            company_row = self.filtered_df[self.filtered_df['Company'] == selected_detail_company].iloc[0]
            company_key = f"{company_row['Ticker']}_{company_row['Quarter']}_{company_row['Year']}"
            
            if company_key in self.results_data:
                st.json(self.results_data[company_key])
    
    def _generate_insights(self) -> Dict[str, str]:
        """Generate key insights from the data"""
        insights = {}
        
        if len(self.filtered_df) == 0:
            return {"No Data": "No data available for analysis."}
        
        # Rating distribution insight
        rating_dist = self.filtered_df['Investment_Rating'].value_counts()
        most_common_rating = rating_dist.index[0]
        insights["Rating Distribution"] = f"Most common investment rating: {most_common_rating} ({rating_dist[most_common_rating]} companies)"
        
        # Content length insight
        avg_content = self.filtered_df['Content_Length'].mean()
        insights["Content Analysis"] = f"Average transcript length: {avg_content:,.0f} characters. "
        
        if self.filtered_df['Content_Length'].std() > 0:
            longest_company = self.filtered_df.loc[self.filtered_df['Content_Length'].idxmax(), 'Company']
            insights["Content Analysis"] += f"Longest transcript: {longest_company}"
        
        # Q&A insight
        qa_percentage = (self.filtered_df['Has_QA'].sum() / len(self.filtered_df)) * 100
        insights["Q&A Sessions"] = f"{qa_percentage:.1f}% of transcripts include Q&A sessions"
        
        # Temporal insight
        if 'Year' in self.filtered_df.columns and len(self.filtered_df['Year'].unique()) > 1:
            year_counts = self.filtered_df['Year'].value_counts()
            most_active_year = year_counts.index[0]
            insights["Temporal Analysis"] = f"Most active year: {most_active_year} ({year_counts[most_active_year]} companies)"
        
        return insights
    
    def run_new_analysis(self):
        """Run new analysis on available transcripts"""
        with st.spinner("Running new analysis..."):
            try:
                self.orchestrator = TranscriptAnalysisOrchestrator()
                results = self.orchestrator.analyze_all_transcripts()
                
                if results:
                    st.success("✅ Analysis completed successfully!")
                    st.rerun()
                else:
                    st.error("❌ Analysis failed. Please check your configuration and data.")
            
            except Exception as e:
                st.error(f"❌ Error running analysis: {str(e)}")

def main():
    """Main function to run the Streamlit dashboard"""
    dashboard = InvestmentDashboard()
    dashboard.render_dashboard()

if __name__ == "__main__":
    main()