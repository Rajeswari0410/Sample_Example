#!/usr/bin/env python3
"""
Quick start script for Earnings Call Investment Insights System
"""

import argparse
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    try:
        from config import validate_config
        validate_config()
        print("✅ Configuration validated successfully")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        print("Please ensure you have:")
        print("1. Set OPENAI_API_KEY in your .env file")
        print("2. Installed all requirements: pip install -r requirements.txt")
        return False

def run_analysis():
    """Run the full transcript analysis"""
    try:
        from crew_orchestrator import main
        print("🚀 Starting transcript analysis...")
        main()
        return True
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    try:
        import subprocess
        print("🖥️ Launching dashboard...")
        subprocess.run(["streamlit", "run", "streamlit_dashboard.py"])
        return True
    except Exception as e:
        print(f"❌ Dashboard launch failed: {e}")
        print("Please ensure Streamlit is installed: pip install streamlit")
        return False

def create_sample_data():
    """Create sample transcript data for testing"""
    print("📁 Sample data already exists in MyDrive/Earnings2Insights/ECTsum/")
    print("You can add more transcript files following the same structure:")
    print("  MyDrive/Earnings2Insights/ECTsum/TICKER_qX_YEAR/source/source.md")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Earnings Call Investment Insights System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py --check          # Check configuration
  python run_analysis.py --analyze        # Run analysis
  python run_analysis.py --dashboard      # Launch dashboard
  python run_analysis.py --sample         # Show sample data info
        """
    )
    
    parser.add_argument(
        "--check", 
        action="store_true", 
        help="Check configuration and requirements"
    )
    
    parser.add_argument(
        "--analyze", 
        action="store_true", 
        help="Run comprehensive transcript analysis"
    )
    
    parser.add_argument(
        "--dashboard", 
        action="store_true", 
        help="Launch the interactive Streamlit dashboard"
    )
    
    parser.add_argument(
        "--sample", 
        action="store_true", 
        help="Show information about sample data structure"
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    print("📊 Earnings Call Investment Insights System")
    print("=" * 50)
    
    if args.check:
        if check_requirements():
            print("🎉 System is ready to use!")
        else:
            sys.exit(1)
    
    if args.sample:
        create_sample_data()
    
    if args.analyze:
        if not check_requirements():
            sys.exit(1)
        
        if not run_analysis():
            sys.exit(1)
        
        print("\n🎉 Analysis completed!")
        print("💡 Next steps:")
        print("  1. Review results in the 'analysis_results' directory")
        print("  2. Launch dashboard: python run_analysis.py --dashboard")
    
    if args.dashboard:
        if not os.path.exists("analysis_results"):
            print("⚠️ No analysis results found.")
            print("Run analysis first: python run_analysis.py --analyze")
            return
        
        launch_dashboard()

if __name__ == "__main__":
    main()