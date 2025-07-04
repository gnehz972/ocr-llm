#!/usr/bin/env python3
"""
Batch OCR using Google Gemini 2.0 Flash API with Google Gen AI SDK
"""

from google import genai
import os
import argparse
import json
from pathlib import Path
from PIL import Image
import time
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class GeminiBatchOCR:
    def __init__(self, api_key=None):
        """Initialize Gemini API client"""
        api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Please provide Gemini API key via --api-key or GEMINI_API_KEY environment variable")
        
        # Use Gemini 2.0 Flash model with new SDK
        self.client = genai.Client(api_key=api_key)
        self.model_name = 'gemini-2.0-flash-exp'
        
    def process_single_image(self, image_path, prompt="Extract all text from this image, do not add any additional text"):
        """Process a single image with OCR"""
        try:
            # Load and process image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Generate content using new SDK
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image]
            )
            
            return {
                'image_path': str(image_path),
                'success': True,
                'text': response.text,
                'error': None
            }
            
        except Exception as e:
            return {
                'image_path': str(image_path),
                'success': False,
                'text': None,
                'error': str(e)
            }
    
    
    def batch_process(self, image_paths, prompt="Extract all text from this image", delay=1.0):
        """Process multiple images with OCR"""
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing {i+1}/{len(image_paths)}: {image_path}")
            
            result = self.process_single_image(image_path, prompt)
            results.append(result)
            
            if result['success']:
                print(f"✓ Extracted {len(result['text'])} characters")
                print(f"Response: {result['text'][:200]}..." if len(result['text']) > 200 else f"Response: {result['text']}")
            else:
                print(f"✗ Error: {result['error']}")
            
            # Rate limiting delay
            if i < len(image_paths) - 1:
                time.sleep(delay)
        
        return results
    
    def save_results(self, results, output_file):
        """Save results to file"""
        if output_file.endswith('.json'):
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        elif output_file.endswith('.md'):
            self.save_markdown_table(results, output_file)
        else:
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    if result['success']:
                        f.write(f"{result['text']}\n\n")
        
        print(f"Results saved to: {output_file}")
    
    def save_markdown_table(self, results, output_file):
        """Save results as markdown table"""
        with open(output_file, 'w', encoding='utf-8') as f:
            # Combine all extracted text into one table
            all_text = []
            for result in results:
                if result['success']:
                    all_text.append(result['text'])
            
            if all_text:
                # Parse and merge markdown tables
                merged_table = self.merge_markdown_tables(all_text)
                f.write(merged_table)
    
    def merge_markdown_tables(self, text_list):
        """Merge multiple markdown tables into one"""
        all_rows = []
        header_written = False
        
        for text in text_list:
            lines = text.strip().split('\n')
            table_rows = []
            
            for line in lines:
                line = line.strip()
                # Skip markdown code blocks
                if line.startswith('```'):
                    continue
                # Process table rows
                if line.startswith('|') and line.endswith('|'):
                    table_rows.append(line)
            
            # Add header from first table only
            if table_rows and not header_written:
                # Add the first row as header
                if len(table_rows) > 0:
                    all_rows.append(table_rows[0])  # Header row
                if len(table_rows) > 1:
                    all_rows.append(table_rows[1])  # Separator row
                    header_written = True
                # Add data rows (skip header and separator)
                for row in table_rows[2:]:
                    if self.is_valid_data_row(row):
                        all_rows.append(row)
            else:
                # For subsequent tables, skip header and separator, add only data rows
                for row in table_rows:
                    # Skip header-like rows and separators
                    if self.is_valid_data_row(row):
                        all_rows.append(row)
        
        return '\n'.join(all_rows)
    
    def is_valid_data_row(self, row):
        """Check if a table row contains valid data (not empty or whitespace only)"""
        # Split the row by | and check if there's actual content
        cells = [cell.strip() for cell in row.split('|')[1:-1]]  # Remove first and last empty elements
        
        # Check if all cells have content (not empty or whitespace only)
        valid_cells = [cell for cell in cells if cell and not cell.isspace()]
        
        # Skip if any cell is empty or if it's a header row
        if len(valid_cells) != len(cells):
            return False

        # Skip header-like rows
        if any(header_word in row for header_word in ['分数', 'Score', 'Rank', 'ID', '值', '数量', '-', '---']):
            return False
            
        return True

def find_image_files(directory):
    """Find all image files in directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_files = []
    
    for file_path in Path(directory).rglob('*'):
        if file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    return sorted(image_files)

def main():
    parser = argparse.ArgumentParser(description='Batch OCR using Gemini 2.0 Flash API with Google Gen AI SDK')
    parser.add_argument('images', nargs='*', help='Image file paths or directory')
    parser.add_argument('--api-key', help='Gemini API key (or set GEMINI_API_KEY env var)')
    parser.add_argument('--prompt', default='Extract structured text from this image and return it in markdown format, do not add any additional text',
                       help='OCR prompt to use')
    parser.add_argument('--output', '-o', help='Output file for results')
    parser.add_argument('--delay', type=float, default=1.0, 
                       help='Delay between API calls (seconds)')
    parser.add_argument('--directory', '-d', help='Process all images in directory')
    
    args = parser.parse_args()
    
    # Collect image files
    image_files = []
    
    if args.directory:
        image_files.extend(find_image_files(args.directory))
    
    for img_path in args.images:
        path = Path(img_path)
        if path.is_file():
            image_files.append(path)
        elif path.is_dir():
            image_files.extend(find_image_files(path))
    
    if not image_files:
        print("No image files found!")
        sys.exit(1)
    
    print(f"Found {len(image_files)} image files")
    
    # Initialize OCR processor
    try:
        ocr = GeminiBatchOCR(args.api_key)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Process images
    results = ocr.batch_process(image_files, args.prompt, args.delay)
    
    # Save results
    if args.output:
        ocr.save_results(results, args.output)
    else:
        # Print results to console
        for result in results:
            print(f"\n{'='*50}")
            print(f"Image: {result['image_path']}")
            print('='*50)
            if result['success']:
                print(result['text'])
            else:
                print(f"ERROR: {result['error']}")
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"\nSummary: {successful}/{len(results)} images processed successfully")

if __name__ == "__main__":
    main()