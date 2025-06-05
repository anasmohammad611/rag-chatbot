# backend/scraper.py
import json
import logging
import re
import time
from typing import List, Dict, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AngelOneScraper:
    def __init__(self):
        self.base_url = "https://www.angelone.in/support"
        self.visited_urls: Set[str] = set()
        self.scraped_data: List[Dict] = []
        self.session = requests.Session()

        # Set headers to avoid being blocked
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def is_valid_support_url(self, url: str) -> bool:
        """Check if URL is a valid AngelOne support page"""
        parsed = urlparse(url)
        return (
                parsed.netloc == "www.angelone.in" and
                "/support" in parsed.path and
                url not in self.visited_urls and
                "#" not in url  # Skip anchor links
        )

    def clean_extracted_text(self, text: str) -> str:
        """Enhanced text cleaning to remove navigation and improve quality"""
        if not text:
            return ""

        logger.info("🧹 [CLEAN] Starting enhanced text cleaning...")

        # Step 1: Remove common navigation patterns
        navigation_patterns = [
            r'Quick Links.*?Learn More',
            r'We are here to help you.*?Learn More',
            r'Track Application Status.*?Learn More',
            r'Learn More\s*',
            r'Know.*?Learn More',
            r'Still need help\?.*?Connect with us',
            r'Want to connect with us\?.*?Connect with us',
            r'Our experts will be happy to assist you.*?Connect with us',
            r'Create Ticket.*?Connect with us',
            r'Still have any queries\?.*?Connect with us',
            r'\(\d+\)\s*',  # Remove numbers in parentheses like (10)
            r'En\s*$',  # Remove trailing "En"
        ]

        for pattern in navigation_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

        # Step 2: Remove repetitive phrases
        repetitive_phrases = [
            "We are here to help you",
            "Learn More",
            "Know how to",
            "Learn how to",
            "Know all about",
            "Step by step guide",
            "Know the process",
            "Track Application Status",
            "Create Ticket",
            "Connect with us",
            "Still need help?",
            "Want to connect with us?",
            "Our experts will be happy to assist you",
            "Still have any queries?"
        ]

        for phrase in repetitive_phrases:
            text = text.replace(phrase, '')

        # Step 3: Fix spacing and formatting
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\n+', ' ', text)  # Multiple newlines to space
        text = text.strip()

        # Step 4: Remove very short segments (likely navigation fragments)
        sentences = text.split('.')
        meaningful_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and not self.is_navigation_text(sentence):
                meaningful_sentences.append(sentence)

        # Step 5: Reconstruct text from meaningful sentences
        cleaned_text = '. '.join(meaningful_sentences)
        if cleaned_text and not cleaned_text.endswith('.'):
            cleaned_text += '.'

        logger.info(f"🧹 [CLEAN] Original length: {len(text)} chars")
        logger.info(f"🧹 [CLEAN] Cleaned length: {len(cleaned_text)} chars")
        logger.info(f"🧹 [CLEAN] Reduction: {((len(text) - len(cleaned_text)) / len(text) * 100):.1f}%")

        return cleaned_text

    def is_navigation_text(self, text: str) -> bool:
        """Check if text segment is likely navigation/menu content"""
        text_lower = text.lower()

        navigation_indicators = [
            'quick links',
            'learn more',
            'track application',
            'know how to',
            'step by step guide',
            'create ticket',
            'connect with us',
            'our experts',
            'still need help',
            'want to connect'
        ]

        # If text is very short or contains navigation indicators
        if len(text) < 25 or any(indicator in text_lower for indicator in navigation_indicators):
            return True

        # Check ratio of common words vs content words
        common_words = ['the', 'to', 'and', 'a', 'of', 'in', 'is', 'for', 'on', 'with', 'as', 'by']
        words = text_lower.split()
        if len(words) > 0:
            common_ratio = sum(1 for word in words if word in common_words) / len(words)
            if common_ratio > 0.7:  # Too many common words, likely navigation
                return True

        return False

    def get_page_content(self, url: str) -> Dict:
        """Scrape content from a single page with enhanced content extraction"""
        try:
            logger.info(f"🔍 [SCRAPE] Processing: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract title
            title = soup.find('title')
            title_text = title.text.strip() if title else "No Title"
            logger.info(f"🔍 [SCRAPE] Page title: {title_text}")

            # Enhanced removal of non-content elements
            elements_to_remove = [
                'script', 'style', 'nav', 'footer', 'header', 'aside',
                '[role="navigation"]', '.nav', '.navbar', '.menu', '.breadcrumb',
                '.sidebar', '.widget', '.advertisement', '.ad', '.banner',
                '.social-media', '.share-buttons', '.related-links',
                '.quick-links', '.footer-links', '.header-links'
            ]

            for selector in elements_to_remove:
                for element in soup.select(selector):
                    element.decompose()

            logger.info(f"🔍 [SCRAPE] Removed navigation elements")

            # Try to find the main content with priority order
            content_selectors = [
                'main',
                '[role="main"]',
                '.main-content',
                '.content-area',
                '.article-content',
                '.post-content',
                '.page-content',
                '.support-content',
                'article',
                '.content',
                '#content',
                'body'
            ]

            content_text = ""
            content_source = "unknown"

            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Get text but preserve some structure
                    content_text = content_elem.get_text(separator=' ', strip=True)
                    content_source = selector
                    logger.info(f"🔍 [SCRAPE] Content extracted from: {selector}")
                    break

            # Fallback: get all remaining text
            if not content_text or len(content_text.split()) < 50:
                content_text = soup.get_text(separator=' ', strip=True)
                content_source = "fallback"
                logger.info(f"🔍 [SCRAPE] Using fallback text extraction")

            # Enhanced text cleaning
            original_length = len(content_text)
            content_text = self.clean_extracted_text(content_text)

            # Quality check
            word_count = len(content_text.split())
            logger.info(f"🔍 [SCRAPE] Content quality:")
            logger.info(f"🔍 [SCRAPE]   Original length: {original_length} chars")
            logger.info(f"🔍 [SCRAPE]   Cleaned length: {len(content_text)} chars")
            logger.info(f"🔍 [SCRAPE]   Word count: {word_count}")
            logger.info(f"🔍 [SCRAPE]   Content source: {content_source}")

            # Find all links on this page
            links = []
            for link in soup.find_all('a', href=True):
                full_url = urljoin(url, link['href'])
                if self.is_valid_support_url(full_url):
                    links.append(full_url)

            return {
                'url': url,
                'title': title_text,
                'content': content_text,
                'links': links,
                'word_count': word_count,
                'content_source': content_source,
                'scraped_at': time.time()
            }

        except Exception as e:
            logger.error(f"🔍 [SCRAPE] ❌ Error scraping {url}: {str(e)}")
            return None

    def scrape_support_pages(self, max_pages: int = 50) -> List[Dict]:
        """Scrape AngelOne support pages recursively"""
        logger.info(f"🚀 [SCRAPE] Starting enhanced scraping (max {max_pages} pages)")

        # Start with known support URLs
        known_support_urls = [
            "https://www.angelone.in/support",
            "https://www.angelone.in/support/add-and-withdraw-funds",
            "https://www.angelone.in/support/charges-and-cashbacks",
            "https://www.angelone.in/support/your-account",
            "https://www.angelone.in/support/your-orders",
            "https://www.angelone.in/support/margin-pledging-and-margin-trading-facility",
            "https://www.angelone.in/support/mutual-funds",
            "https://www.angelone.in/support/ipo-ofs",
            "https://www.angelone.in/support/reports-and-statements",
            "https://www.angelone.in/support/portfolio-and-corporate-actions"
        ]

        urls_to_visit = known_support_urls.copy()
        successful_pages = 0
        skipped_pages = 0

        while urls_to_visit and len(self.scraped_data) < max_pages:
            current_url = urls_to_visit.pop(0)

            if current_url in self.visited_urls:
                continue

            self.visited_urls.add(current_url)

            # Get page content
            page_data = self.get_page_content(current_url)

            if page_data and page_data['content']:
                # Enhanced quality filtering
                word_count = page_data['word_count']
                content_length = len(page_data['content'])

                # Only save pages with substantial, meaningful content
                if word_count > 100 and content_length > 500:  # Increased thresholds
                    self.scraped_data.append(page_data)
                    successful_pages += 1
                    logger.info(f"✅ [SCRAPE] Saved: {page_data['title']}")
                    logger.info(f"✅ [SCRAPE]   Words: {word_count}, Source: {page_data['content_source']}")
                    logger.info(f"✅ [SCRAPE]   Preview: {page_data['content'][:100]}...")
                else:
                    skipped_pages += 1
                    logger.warning(f"⚠️ [SCRAPE] Skipped low-quality page: {page_data['title']}")
                    logger.warning(f"⚠️ [SCRAPE]   Reason: {word_count} words, {content_length} chars")

                # Add new links to visit (skip duplicates)
                for link in page_data['links']:
                    if (link not in self.visited_urls and
                            link not in urls_to_visit and
                            self.is_valid_support_url(link)):
                        urls_to_visit.append(link)

            time.sleep(1)

        logger.info(f"🎉 [SCRAPE] ✅ Enhanced scraping complete!")
        logger.info(f"🎉 [SCRAPE] Successfully processed: {successful_pages} pages")
        logger.info(f"🎉 [SCRAPE] Skipped low-quality: {skipped_pages} pages")
        logger.info(f"🎉 [SCRAPE] Total URLs visited: {len(self.visited_urls)}")

        return self.scraped_data

    def save_to_file(self, filename: str = "angelone_support_data.json"):
        """Save scraped data to JSON file"""
        logger.info(f"💾 [SAVE] Saving {len(self.scraped_data)} high-quality documents to {filename}")

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, indent=2, ensure_ascii=False)

        # Generate summary statistics
        total_words = sum(doc['word_count'] for doc in self.scraped_data)
        avg_words = total_words / len(self.scraped_data) if self.scraped_data else 0

        logger.info(f"💾 [SAVE] ✅ Data saved successfully!")
        logger.info(f"💾 [SAVE] Summary statistics:")
        logger.info(f"💾 [SAVE]   Total documents: {len(self.scraped_data)}")
        logger.info(f"💾 [SAVE]   Total words: {total_words:,}")
        logger.info(f"💾 [SAVE]   Average words per document: {avg_words:.0f}")


# Test function
def test_scraper():
    """Test the enhanced scraper"""
    logger.info("🧪 [TEST] Starting enhanced scraper test...")

    scraper = AngelOneScraper()

    # Test with more pages to ensure quality
    data = scraper.scrape_support_pages(max_pages=100)

    print(f"\n📊 ENHANCED SCRAPING RESULTS:")
    print(f"Total high-quality pages scraped: {len(data)}")
    print(f"Total words extracted: {sum(doc['word_count'] for doc in data):,}")

    for i, page in enumerate(data[:5], 1):  # Show first 5 pages
        print(f"\n{i}. {page['title']}")
        print(f"   URL: {page['url']}")
        print(f"   Words: {page['word_count']}")
        print(f"   Source: {page['content_source']}")
        print(f"   Quality preview: {page['content'][:150]}...")

    # Save the enhanced data
    scraper.save_to_file()

    # Quality analysis
    print(f"\n🔍 QUALITY ANALYSIS:")
    word_counts = [doc['word_count'] for doc in data]
    if word_counts:
        print(f"   Min words: {min(word_counts)}")
        print(f"   Max words: {max(word_counts)}")
        print(f"   Avg words: {sum(word_counts) / len(word_counts):.0f}")

    return data


if __name__ == "__main__":
    test_scraper()
