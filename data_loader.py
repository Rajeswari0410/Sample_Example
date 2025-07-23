import os
import re
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TranscriptData:
    """Data class to hold transcript information"""
    company_name: str
    ticker: str
    quarter: str
    year: str
    file_path: str
    content: str
    prepared_remarks: str
    qa_section: str
    
class TranscriptLoader:
    """Handles loading and parsing of earnings call transcripts"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.transcripts = []
        
    def load_all_transcripts(self) -> List[TranscriptData]:
        """Load all transcript files from the directory structure"""
        transcript_files = []
        
        # Walk through directory structure to find source.md files
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file == "source.md":
                    file_path = os.path.join(root, file)
                    transcript_files.append(file_path)
        
        print(f"Found {len(transcript_files)} transcript files")
        
        for file_path in transcript_files:
            try:
                transcript = self._parse_transcript_file(file_path)
                if transcript:
                    self.transcripts.append(transcript)
            except Exception as e:
                print(f"Error parsing {file_path}: {str(e)}")
                
        return self.transcripts
    
    def _parse_transcript_file(self, file_path: str) -> Optional[TranscriptData]:
        """Parse individual transcript file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract company info from file path
            path_parts = Path(file_path).parts
            company_info = self._extract_company_info_from_path(path_parts)
            
            # Split content into sections
            prepared_remarks, qa_section = self._split_transcript_sections(content)
            
            return TranscriptData(
                company_name=company_info['company_name'],
                ticker=company_info['ticker'],
                quarter=company_info['quarter'],
                year=company_info['year'],
                file_path=file_path,
                content=content,
                prepared_remarks=prepared_remarks,
                qa_section=qa_section
            )
            
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return None
    
    def _extract_company_info_from_path(self, path_parts: tuple) -> Dict[str, str]:
        """Extract company information from file path"""
        # Example path: MyDrive/Earnings2Insights/ECTsum/ABM_q3_2021/source/source.md
        company_folder = None
        
        for part in path_parts:
            if '_q' in part and '_20' in part:  # Pattern like ABM_q3_2021
                company_folder = part
                break
        
        if company_folder:
            # Parse company_quarter_year pattern
            parts = company_folder.split('_')
            if len(parts) >= 3:
                ticker = parts[0]
                quarter = parts[1]  # e.g., 'q3'
                year = parts[2]     # e.g., '2021'
                
                # Convert ticker to company name (you might want to use a mapping)
                company_name = self._ticker_to_company_name(ticker)
                
                return {
                    'company_name': company_name,
                    'ticker': ticker.upper(),
                    'quarter': quarter.upper(),
                    'year': year
                }
        
        # Fallback
        return {
            'company_name': 'Unknown',
            'ticker': 'UNK',
            'quarter': 'Q1',
            'year': '2021'
        }
    
    def _ticker_to_company_name(self, ticker: str) -> str:
        """Convert ticker symbol to company name"""
        # Basic mapping - you can expand this
        ticker_mapping = {
            'ABM': 'ABM Industries',
            'AME': 'AMETEK Inc',
            'CFR': 'Cullen/Frost Bankers'
        }
        return ticker_mapping.get(ticker.upper(), ticker.upper())
    
    def _split_transcript_sections(self, content: str) -> tuple:
        """Split transcript into prepared remarks and Q&A sections"""
        # Look for Q&A section marker
        qa_pattern = r'###\s*Q&A'
        match = re.search(qa_pattern, content, re.IGNORECASE)
        
        if match:
            split_index = match.start()
            prepared_remarks = content[:split_index].strip()
            qa_section = content[split_index:].strip()
        else:
            # If no Q&A section found, treat entire content as prepared remarks
            prepared_remarks = content
            qa_section = ""
        
        return prepared_remarks, qa_section
    
    def get_transcript_by_ticker(self, ticker: str) -> List[TranscriptData]:
        """Get all transcripts for a specific ticker"""
        return [t for t in self.transcripts if t.ticker.upper() == ticker.upper()]
    
    def get_transcript_summary(self) -> pd.DataFrame:
        """Get summary of all loaded transcripts"""
        if not self.transcripts:
            return pd.DataFrame()
        
        summary_data = []
        for transcript in self.transcripts:
            summary_data.append({
                'Company': transcript.company_name,
                'Ticker': transcript.ticker,
                'Quarter': transcript.quarter,
                'Year': transcript.year,
                'Content Length': len(transcript.content),
                'Has Q&A': len(transcript.qa_section) > 0,
                'File Path': transcript.file_path
            })
        
        return pd.DataFrame(summary_data)

# Utility functions
def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)]', '', text)
    return text.strip()

def extract_speaker_statements(content: str) -> Dict[str, List[str]]:
    """Extract statements by speaker type"""
    statements = {
        'CEO': [],
        'CFO': [],
        'Analyst': [],
        'Other': []
    }
    
    # Pattern to match speaker labels
    speaker_pattern = r'\*\*(.*?)\*\*\s*:\s*(.*?)(?=\*\*|$)'
    matches = re.findall(speaker_pattern, content, re.DOTALL)
    
    for speaker, statement in matches:
        speaker = speaker.strip()
        statement = clean_text(statement)
        
        if 'CEO' in speaker.upper() or 'CHIEF EXECUTIVE' in speaker.upper():
            statements['CEO'].append(statement)
        elif 'CFO' in speaker.upper() or 'CHIEF FINANCIAL' in speaker.upper():
            statements['CFO'].append(statement)
        elif 'ANALYST' in speaker.upper():
            statements['Analyst'].append(statement)
        else:
            statements['Other'].append(statement)
    
    return statements