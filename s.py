import streamlit as st
import PyPDF2
import docx
import io
import re
from typing import Optional, Dict, List, Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import heapq
import traceback
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import numpy as np
from textstat import flesch_reading_ease, flesch_kincaid_grade
import math

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data with caching"""
    try:
        # Try the new punkt_tab first
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            # Try the old punkt
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                # Download punkt_tab (newer version)
                nltk.download('punkt_tab')
            except:
                try:
                    # Fallback to punkt (older version)
                    nltk.download('punkt')
                except:
                    st.warning("Could not download punkt tokenizer. Some features may not work properly.")
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords')
        except:
            st.warning("Could not download stopwords. Using basic stopwords list.")

class DocumentSummarizer:
    def __init__(self):
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback stopwords if NLTK download fails
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once'
            }
        self.custom_stop_words = {'said', 'says', 'mr', 'mrs', 'ms', 'dr', 'prof'}
        self.all_stop_words = self.stop_words.union(self.custom_stop_words)
    
    def safe_sent_tokenize(self, text: str) -> List[str]:
        """Safe sentence tokenization with fallback"""
        try:
            return sent_tokenize(text)
        except:
            # Fallback: simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def safe_word_tokenize(self, text: str) -> List[str]:
        """Safe word tokenization with fallback"""
        try:
            return word_tokenize(text)
        except:
            # Fallback: simple word splitting
            words = re.findall(r'\b\w+\b', text.lower())
            return words
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file with improved error handling"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            total_pages = len(pdf_reader.pages)
            
            # Show progress for large PDFs
            if total_pages > 10:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            for i, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    text += page_text + "\n"
                except Exception as e:
                    st.warning(f"Could not extract text from page {i + 1}: {str(e)}")
                    continue
                
                # Update progress for large PDFs
                if total_pages > 10:
                    progress = (i + 1) / total_pages
                    progress_bar.progress(progress)
                    status_text.text(f"Processing page {i + 1} of {total_pages}")
            
            # Clean up progress indicators
            if total_pages > 10:
                progress_bar.empty()
                status_text.empty()
            
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, docx_file) -> str:
        """Extract text from DOCX file with improved formatting"""
        try:
            doc = docx.Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():  # Skip empty paragraphs
                    text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""
    
    def extract_text_from_txt(self, txt_file) -> str:
        """Extract text from TXT file with encoding detection"""
        try:
            # Try UTF-8 first
            txt_file.seek(0)
            content = txt_file.read()
            
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    if isinstance(content, bytes):
                        return content.decode(encoding)
                    else:
                        return str(content)
                except (UnicodeDecodeError, AttributeError):
                    continue
            
            # Fallback
            return str(content, errors='ignore')
        except Exception as e:
            st.error(f"Error reading TXT: {str(e)}")
            return ""
    
    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing"""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep periods, commas, and exclamation marks
        text = re.sub(r'[^\w\s\.\,\!\?\:\;]', '', text)
        # Remove multiple punctuation marks
        text = re.sub(r'[\.]{2,}', '.', text)
        # Fix spacing around punctuation
        text = re.sub(r'\s+([\.!\?])', r'\1', text)
        return text.strip()
    
    def calculate_word_frequencies(self, text: str) -> Dict[str, int]:
        """Calculate word frequencies with improved filtering"""
        words = self.safe_word_tokenize(text.lower())
        words = [self.stemmer.stem(word) for word in words 
                if word not in self.all_stop_words and word.isalpha() and len(word) > 2]
        return Counter(words)
    
    def calculate_sentence_position_score(self, sentences: List[str], position: int) -> float:
        """Calculate position-based score (first and last sentences are more important)"""
        total_sentences = len(sentences)
        if total_sentences == 1:
            return 1.0
        
        # Give higher scores to sentences at the beginning and end
        if position == 0:
            return 1.0
        elif position == total_sentences - 1:
            return 0.8
        elif position < total_sentences * 0.1:  # First 10%
            return 0.9
        elif position > total_sentences * 0.9:  # Last 10%
            return 0.7
        else:
            return 0.5
    
    def calculate_sentence_length_score(self, sentence: str) -> float:
        """Calculate length-based score (prefer moderate length sentences)"""
        word_count = len(self.safe_word_tokenize(sentence))
        if 10 <= word_count <= 30:
            return 1.0
        elif 5 <= word_count < 10 or 30 < word_count <= 50:
            return 0.8
        else:
            return 0.6
    
    def score_sentences(self, sentences: List[str], word_freq: Dict[str, int]) -> Dict[str, float]:
        """Enhanced sentence scoring with multiple factors"""
        sentence_scores = {}
        
        for i, sentence in enumerate(sentences):
            words = self.safe_word_tokenize(sentence.lower())
            words = [self.stemmer.stem(word) for word in words 
                    if word not in self.all_stop_words and word.isalpha() and len(word) > 2]
            
            # Base frequency score
            freq_score = 0
            word_count = len(words)
            
            if word_count > 0:
                for word in words:
                    if word in word_freq:
                        freq_score += word_freq[word]
                freq_score = freq_score / word_count
            
            # Position score
            position_score = self.calculate_sentence_position_score(sentences, i)
            
            # Length score
            length_score = self.calculate_sentence_length_score(sentence)
            
            # Combined score with weights
            final_score = (freq_score * 0.6) + (position_score * 0.3) + (length_score * 0.1)
            sentence_scores[sentence] = final_score
                
        return sentence_scores
    
    def extractive_summarize(self, text: str, num_sentences: int = 3, algorithm: str = "enhanced") -> str:
        """Create extractive summary with multiple algorithm options"""
        if not text or len(text.strip()) == 0:
            return "No text to summarize."
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Tokenize into sentences
        sentences = self.safe_sent_tokenize(processed_text)
        
        if len(sentences) <= num_sentences:
            return processed_text
        
        # Calculate word frequencies
        word_freq = self.calculate_word_frequencies(processed_text)
        
        # Score sentences based on algorithm choice
        if algorithm == "enhanced":
            sentence_scores = self.score_sentences(sentences, word_freq)
        else:  # basic algorithm
            sentence_scores = {}
            for sentence in sentences:
                words = self.safe_word_tokenize(sentence.lower())
                words = [self.stemmer.stem(word) for word in words 
                        if word not in self.all_stop_words and word.isalpha()]
                
                score = 0
                word_count = len(words)
                
                if word_count > 0:
                    for word in words:
                        if word in word_freq:
                            score += word_freq[word]
                    sentence_scores[sentence] = score / word_count
                else:
                    sentence_scores[sentence] = 0
        
        # Get top sentences
        top_sentences = heapq.nlargest(num_sentences, sentence_scores, 
                                     key=sentence_scores.get)
        
        # Sort sentences by their original order
        summary_sentences = []
        for sentence in sentences:
            if sentence in top_sentences:
                summary_sentences.append(sentence)
        
        return ' '.join(summary_sentences)
    
    def get_document_stats(self, text: str) -> Dict[str, any]:
        """Get comprehensive document statistics"""
        sentences = self.safe_sent_tokenize(text)
        words = self.safe_word_tokenize(text)
        
        # Calculate readability scores
        try:
            flesch_score = flesch_reading_ease(text)
            fk_grade = flesch_kincaid_grade(text)
        except:
            flesch_score = 0
            fk_grade = 0
        
        # Calculate average sentence length
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Calculate unique word ratio
        unique_words = len(set(word.lower() for word in words if word.isalpha()))
        unique_ratio = unique_words / len(words) if words else 0
        
        # Estimate reading time (average 200 words per minute)
        reading_time = math.ceil(len(words) / 200) if words else 0
        
        return {
            "characters": len(text),
            "words": len(words),
            "sentences": len(sentences),
            "paragraphs": len([p for p in text.split('\n\n') if p.strip()]),
            "unique_words": unique_words,
            "unique_ratio": unique_ratio,
            "avg_sentence_length": avg_sentence_length,
            "flesch_score": flesch_score,
            "fk_grade": fk_grade,
            "reading_time": reading_time
        }
    
    def get_word_frequency_chart(self, text: str, top_n: int = 10):
        """Generate word frequency chart"""
        word_freq = self.calculate_word_frequencies(text)
        top_words = dict(word_freq.most_common(top_n))
        
        if not top_words:
            return None
        
        df = pd.DataFrame(list(top_words.items()), columns=['Word', 'Frequency'])
        
        fig = px.bar(df, x='Word', y='Frequency', 
                    title=f'Top {top_n} Most Frequent Words',
                    color='Frequency',
                    color_continuous_scale='viridis')
        
        fig.update_layout(
            xaxis_title="Words",
            yaxis_title="Frequency",
            height=400
        )
        
        return fig

def main():
    try:
        st.set_page_config(
            page_title="Enhanced Document Summarizer",
            page_icon="📄",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Download NLTK data
        download_nltk_data()
        
        st.title("📄 Enhanced Document Summarizer")
        st.markdown("Upload your documents and get intelligent summaries with advanced analytics!")
        
        # Initialize the summarizer
        summarizer = DocumentSummarizer()
        
        # Sidebar for settings
        with st.sidebar:
            st.header("⚙️ Settings")
            
            # Summary settings
            st.subheader("Summary Configuration")
            num_sentences = st.slider("Summary Length (sentences)", 
                                     min_value=1, max_value=15, value=3)
            
            algorithm = st.selectbox(
                "Summarization Algorithm",
                ["enhanced", "basic"],
                help="Enhanced: Uses position and length scoring. Basic: Frequency-based only."
            )
            
            # Display options
            st.subheader("Display Options")
            show_stats = st.checkbox("Show Document Statistics", value=True)
            show_chart = st.checkbox("Show Word Frequency Chart", value=True)
            show_readability = st.checkbox("Show Readability Metrics", value=True)
            
            st.divider()
            
            st.header("📋 Supported Formats")
            st.write("• PDF (.pdf)")
            st.write("• Word Documents (.docx)")
            st.write("• Text Files (.txt)")
            
            st.divider()
            
            st.header("ℹ️ Tips")
            st.write("• For best results, use documents with clear paragraph structure")
            st.write("• Longer documents typically produce better summaries")
            st.write("• Try different summary lengths to find optimal results")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a document to summarize",
            type=['pdf', 'docx', 'txt'],
            help="Upload a PDF, Word document, or text file (max 200MB)"
        )
        
        if uploaded_file is not None:
            # Create tabs for better organization
            tab1, tab2, tab3 = st.tabs(["📊 Document Analysis", "📝 Summary", "📈 Analytics"])
            
            # Extract text based on file type
            text = ""
            with st.spinner("Extracting text from document..."):
                try:
                    if uploaded_file.type == "application/pdf":
                        text = summarizer.extract_text_from_pdf(uploaded_file)
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        text = summarizer.extract_text_from_docx(uploaded_file)
                    elif uploaded_file.type == "text/plain":
                        text = summarizer.extract_text_from_txt(uploaded_file)
                    else:
                        st.error("Unsupported file type")
                        return
                except Exception as e:
                    st.error(f"Error extracting text: {str(e)}")
                    return
            
            if not text or len(text.strip()) < 10:
                st.error("Could not extract meaningful text from the document. Please check the file and try again.")
                return
            
            # Get document statistics
            stats = summarizer.get_document_stats(text)
            
            # Tab 1: Document Analysis
            with tab1:
                st.header("📊 Document Information")
                
                # File details
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("File Details")
                    st.write(f"**Filename:** {uploaded_file.name}")
                    st.write(f"**File Type:** {uploaded_file.type}")
                    st.write(f"**File Size:** {uploaded_file.size:,} bytes")
                    st.write(f"**Upload Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                with col2:
                    st.subheader("Text Statistics")
                    st.metric("Characters", f"{stats['characters']:,}")
                    st.metric("Words", f"{stats['words']:,}")
                    st.metric("Sentences", f"{stats['sentences']:,}")
                    st.metric("Paragraphs", f"{stats['paragraphs']:,}")
                
                if show_stats:
                    st.subheader("📈 Advanced Statistics")
                    col3, col4, col5 = st.columns(3)
                    
                    with col3:
                        st.metric("Unique Words", f"{stats['unique_words']:,}")
                        st.metric("Vocabulary Richness", f"{stats['unique_ratio']:.2%}")
                    
                    with col4:
                        st.metric("Avg Sentence Length", f"{stats['avg_sentence_length']:.1f} words")
                        st.metric("Estimated Reading Time", f"{stats['reading_time']} min")
                    
                    with col5:
                        if show_readability:
                            st.metric("Flesch Reading Ease", f"{stats['flesch_score']:.1f}")
                            st.metric("Grade Level", f"{stats['fk_grade']:.1f}")
                
                # Show original text preview
                with st.expander("📖 Document Preview (First 1000 characters)"):
                    st.text_area("", value=text[:1000] + "..." if len(text) > 1000 else text, 
                               height=200, disabled=True)
            
            # Tab 2: Summary
            with tab2:
                st.header("📝 Document Summary")
                
                # Generate summary
                with st.spinner("Generating summary..."):
                    try:
                        summary = summarizer.extractive_summarize(text, num_sentences, algorithm)
                        summary_stats = summarizer.get_document_stats(summary)
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
                        return
                
                # Display summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Summary Length", f"{summary_stats['words']} words")
                with col2:
                    compression_ratio = (1 - summary_stats['words']/stats['words']) if stats['words'] > 0 else 0
                    st.metric("Compression Ratio", f"{compression_ratio:.1%}")
                with col3:
                    st.metric("Sentences Used", f"{summary_stats['sentences']}")
                with col4:
                    st.metric("Reading Time", f"{summary_stats['reading_time']} min")
                
                # Display summary
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 2px;
                        border-radius: 12px;
                        margin: 20px 0;
                    ">
                        <div style="
                            background-color: white;
                            padding: 25px;
                            border-radius: 10px;
                            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        ">
                            <h3 style="color: #333; margin-top: 0; margin-bottom: 15px;">📋 Summary</h3>
                            <p style="font-size: 16px; line-height: 1.7; color: #555; margin-bottom: 0;">
                                {summary}
                            </p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Summary",
                        data=summary,
                        file_name=f"summary_{uploaded_file.name.split('.')[0]}.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    # Create detailed report
                    report = f"""Document Summary Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Original File: {uploaded_file.name}

SUMMARY:
{summary}

STATISTICS:
- Original Words: {stats['words']:,}
- Summary Words: {summary_stats['words']:,}
- Compression Ratio: {compression_ratio:.1%}
- Algorithm Used: {algorithm}
- Sentences in Summary: {summary_stats['sentences']}
"""
                    st.download_button(
                        label="📊 Download Report",
                        data=report,
                        file_name=f"report_{uploaded_file.name.split('.')[0]}.txt",
                        mime="text/plain"
                    )
            
            # Tab 3: Analytics
            with tab3:
                st.header("📈 Document Analytics")
                
                if show_chart:
                    # Word frequency chart
                    fig = summarizer.get_word_frequency_chart(text)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No word frequency data available")
                
                # Readability analysis
                if show_readability:
                    st.subheader("📚 Readability Analysis")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Flesch Reading Ease interpretation
                        flesch_score = stats['flesch_score']
                        if flesch_score >= 90:
                            level = "Very Easy"
                            color = "green"
                        elif flesch_score >= 80:
                            level = "Easy"
                            color = "lightgreen"
                        elif flesch_score >= 70:
                            level = "Fairly Easy"
                            color = "yellow"
                        elif flesch_score >= 60:
                            level = "Standard"
                            color = "orange"
                        elif flesch_score >= 50:
                            level = "Fairly Difficult"
                            color = "red"
                        else:
                            level = "Very Difficult"
                            color = "darkred"
                        
                        st.markdown(f"""
                        **Flesch Reading Ease:** {flesch_score:.1f}  
                        **Level:** <span style="color: {color}; font-weight: bold;">{level}</span>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        **Grade Level:** {stats['fk_grade']:.1f}  
                        **Avg Sentence Length:** {stats['avg_sentence_length']:.1f} words
                        """)
                
                # Additional statistics
                st.subheader("📊 Content Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    chars_per_word = stats['characters']/stats['words'] if stats['words'] > 0 else 0
                    sentences_per_paragraph = stats['sentences']/stats['paragraphs'] if stats['paragraphs'] > 0 else 0
                    st.markdown(f"""
                    **Text Density:**
                    - Characters per word: {chars_per_word:.1f}
                    - Words per sentence: {stats['avg_sentence_length']:.1f}
                    - Sentences per paragraph: {sentences_per_paragraph:.1f}
                    """)
                
                with col2:
                    st.markdown(f"""
                    **Vocabulary Analysis:**
                    - Unique words: {stats['unique_words']:,}
                    - Vocabulary richness: {stats['unique_ratio']:.2%}
                    - Estimated reading time: {stats['reading_time']} minutes
                    """)
        
        else:
            # Welcome screen
            st.info("👆 Upload a document to get started!")
            
            # Feature showcase
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                ### 🎯 Smart Summarization
                - **Enhanced Algorithm**: Position and length-aware scoring
                - **Customizable Length**: 1-15 sentences
                - **Multiple Formats**: PDF, DOCX, TXT support
                """)
            
            with col2:
                st.markdown("""
                ### 📊 Advanced Analytics
                - **Readability Metrics**: Flesch-Kincaid scoring
                - **Word Frequency**: Visual charts
                - **Document Statistics**: Comprehensive analysis
                """)
            
            with col3:
                st.markdown("""
                ### 💾 Export Options
                - **Download Summary**: Plain text format
                - **Detailed Report**: Complete analysis
                - **Chart Export**: Save visualizations
                """)
            
            # Usage instructions
            st.markdown("""
            ### 🚀 How to Use:
            1. **Upload** your document using the file uploader above
            2. **Configure** summary settings in the sidebar
            3. **Analyze** your document across three tabs:
               - **Document Analysis**: File info and statistics
               - **Summary**: Generated summary with metrics
               - **Analytics**: Charts and readability analysis
            4. **Download** summaries and reports as needed
            """)
    
    except Exception as e:
        st.error(f"A critical error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")
        st.error("If the problem persists, try installing the latest version of NLTK.")

if __name__ == "__main__":
    main()