import json
import logging
import os
import time
from typing import List, Dict

import PyPDF2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AngelOnePDFScraper:
    def __init__(self, pdf_directory: str = "./pdfs"):
        """
        Initialize PDF scraper

        Args:
            pdf_directory: Directory containing PDF files
        """
        self.pdf_directory = pdf_directory
        self.scraped_data: List[Dict] = []

        logger.info(f"📁 [INIT] PDF Scraper initialized")
        logger.info(f"📁 [INIT] Looking for PDFs in: {pdf_directory}")

    def extract_text_from_pdf(self, pdf_path: str) -> Dict:
        """Extract text from a single PDF file"""
        try:
            logger.info(f"📄 [PDF] Processing: {pdf_path}")

            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                # Get PDF metadata
                num_pages = len(pdf_reader.pages)
                logger.info(f"📄 [PDF] Found {num_pages} pages in {os.path.basename(pdf_path)}")

                # Extract text from all pages
                full_text = ""
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        full_text += page_text + "\n"
                        logger.info(f"📄 [PDF] Extracted page {page_num}/{num_pages}")
                    except Exception as e:
                        logger.warning(f"📄 [PDF] Could not extract page {page_num}: {e}")

                # Clean up the text
                full_text = self.clean_text(full_text)

                # Create document structure similar to web scraper
                pdf_document = {
                    'url': f"file://{pdf_path}",  # Local file reference
                    'title': self.extract_title_from_filename(pdf_path),
                    'content': full_text,
                    'source_type': 'pdf',
                    'file_path': pdf_path,
                    'num_pages': num_pages,
                    'word_count': len(full_text.split()),
                    'scraped_at': time.time()
                }

                logger.info(f"📄 [PDF] ✅ Successfully processed {os.path.basename(pdf_path)}")
                logger.info(f"📄 [PDF] Extracted {pdf_document['word_count']} words from {num_pages} pages")

                return pdf_document

        except FileNotFoundError:
            logger.error(f"📄 [PDF] ❌ File not found: {pdf_path}")
            return None
        except Exception as e:
            logger.error(f"📄 [PDF] ❌ Error processing {pdf_path}: {str(e)}")
            return None

    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        logger.info("🧹 [CLEAN] Cleaning extracted text...")

        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Remove common PDF artifacts
        artifacts_to_remove = [
            '\x0c',  # Form feed
            '\uf0b7',  # Bullet points
            '\uf020',  # Space characters
        ]

        for artifact in artifacts_to_remove:
            text = text.replace(artifact, ' ')

        # Fix common OCR issues
        text = text.replace('fi', 'fi')  # Fix ligatures
        text = text.replace('fl', 'fl')

        # Remove excessive spaces again
        text = ' '.join(text.split())

        logger.info(f"🧹 [CLEAN] ✅ Text cleaned, final length: {len(text)} characters")
        return text

    def extract_title_from_filename(self, pdf_path: str) -> str:
        """Extract a meaningful title from PDF filename"""
        filename = os.path.basename(pdf_path)

        # Remove extension
        title = os.path.splitext(filename)[0]

        # Replace underscores and hyphens with spaces
        title = title.replace('_', ' ').replace('-', ' ')

        # Capitalize properly
        title = ' '.join(word.capitalize() for word in title.split())

        # Add AngelOne prefix if not present
        if 'angel' not in title.lower():
            title = f"AngelOne Insurance - {title}"

        logger.info(f"📝 [TITLE] Generated title: {title}")
        return title

    def scrape_all_pdfs(self) -> List[Dict]:
        """Scrape all PDF files in the directory"""
        logger.info(f"🚀 [SCRAPE] Starting PDF scraping from {self.pdf_directory}")

        # Check if directory exists
        if not os.path.exists(self.pdf_directory):
            logger.error(f"📁 [SCRAPE] ❌ Directory not found: {self.pdf_directory}")
            logger.info(f"📁 [SCRAPE] Please create the directory and add PDF files")
            return []

        # Find all PDF files
        pdf_files = []
        for filename in os.listdir(self.pdf_directory):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_directory, filename)
                pdf_files.append(pdf_path)

        logger.info(f"📁 [SCRAPE] Found {len(pdf_files)} PDF files")

        if not pdf_files:
            logger.warning(f"📁 [SCRAPE] ⚠️ No PDF files found in {self.pdf_directory}")
            return []

        # Process each PDF
        successful_extractions = 0
        for pdf_path in pdf_files:
            logger.info(f"📄 [SCRAPE] Processing {os.path.basename(pdf_path)}...")

            pdf_data = self.extract_text_from_pdf(pdf_path)

            if pdf_data and pdf_data['content'].strip():
                # Only save PDFs with substantial content
                if pdf_data['word_count'] > 100:  # Minimum 100 words
                    self.scraped_data.append(pdf_data)
                    successful_extractions += 1
                    logger.info(f"📄 [SCRAPE] ✅ Added {os.path.basename(pdf_path)} to dataset")
                else:
                    logger.warning(
                        f"📄 [SCRAPE] ⚠️ Skipped {os.path.basename(pdf_path)} (too short: {pdf_data['word_count']} words)")
            else:
                logger.warning(f"📄 [SCRAPE] ⚠️ Failed to extract content from {os.path.basename(pdf_path)}")

        logger.info(f"🎉 [SCRAPE] ✅ PDF scraping complete!")
        logger.info(f"🎉 [SCRAPE] Successfully processed {successful_extractions}/{len(pdf_files)} PDF files")

        return self.scraped_data

    def save_to_file(self, filename: str = "angelone_pdf_data.json"):
        """Save scraped PDF data to JSON file"""
        logger.info(f"💾 [SAVE] Saving {len(self.scraped_data)} PDF documents to {filename}")

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 [SAVE] ✅ PDF data saved to {filename}")

    def merge_with_web_data(self, web_data_file: str = "angelone_support_data.json",
                            output_file: str = "angelone_combined_data.json"):
        """Merge PDF data with existing web scraping data"""
        logger.info("🔄 [MERGE] Starting data merge process...")

        # Load existing web data
        try:
            with open(web_data_file, 'r', encoding='utf-8') as f:
                web_data = json.load(f)
            logger.info(f"🔄 [MERGE] Loaded {len(web_data)} web documents from {web_data_file}")
        except FileNotFoundError:
            logger.warning(f"🔄 [MERGE] ⚠️ Web data file not found: {web_data_file}")
            logger.info(f"🔄 [MERGE] Creating new combined dataset with PDF data only")
            web_data = []
        except Exception as e:
            logger.error(f"🔄 [MERGE] ❌ Error loading web data: {e}")
            return False

        # Combine datasets
        combined_data = web_data + self.scraped_data

        logger.info(f"🔄 [MERGE] Combined dataset statistics:")
        logger.info(f"🔄 [MERGE]   Web documents: {len(web_data)}")
        logger.info(f"🔄 [MERGE]   PDF documents: {len(self.scraped_data)}")
        logger.info(f"🔄 [MERGE]   Total documents: {len(combined_data)}")

        # Calculate total word count
        total_words = sum(doc.get('word_count', 0) for doc in combined_data)
        logger.info(f"🔄 [MERGE]   Total words: {total_words:,}")

        # Save combined data
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=2, ensure_ascii=False)

            logger.info(f"🔄 [MERGE] ✅ Combined data saved to {output_file}")
            logger.info(f"🔄 [MERGE] Ready for vector database indexing!")
            return True

        except Exception as e:
            logger.error(f"🔄 [MERGE] ❌ Error saving combined data: {e}")
            return False


def main():
    """Main function to run PDF scraping"""
    logger.info("🌟 [MAIN] Starting AngelOne PDF scraping process...")

    # Initialize scraper
    scraper = AngelOnePDFScraper(pdf_directory="./pdfs")

    # Scrape all PDFs
    pdf_data = scraper.scrape_all_pdfs()

    if not pdf_data:
        logger.warning("🌟 [MAIN] ⚠️ No PDF data extracted. Please check your PDF files.")
        return

    # Save PDF data separately
    scraper.save_to_file("angelone_pdf_data.json")

    # Show summary
    print(f"\n📊 PDF SCRAPING RESULTS:")
    print(f"Total PDFs processed: {len(pdf_data)}")

    for i, doc in enumerate(pdf_data, 1):
        print(f"\n{i}. {doc['title']}")
        print(f"   File: {os.path.basename(doc['file_path'])}")
        print(f"   Pages: {doc['num_pages']}")
        print(f"   Words: {doc['word_count']}")
        print(f"   Content preview: {doc['content'][:100]}...")

    # Merge with web data
    logger.info("🌟 [MAIN] Attempting to merge with web scraping data...")
    merge_success = scraper.merge_with_web_data()

    if merge_success:
        print(f"\n🎉 SUCCESS! Combined dataset created.")
        print(f"📁 Next steps:")
        print(f"1. Use 'angelone_combined_data.json' in your vector_db.py")
        print(f"2. Update vector_db.py to load combined data:")
        print(f"   with open('angelone_combined_data.json', 'r') as f:")
        print(f"3. Rebuild your vector database with: python vector_db.py")
    else:
        print(f"\n⚠️ Merge failed. You can manually combine the files or run web scraper first.")


if __name__ == "__main__":
    main()
