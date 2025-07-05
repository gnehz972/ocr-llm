#!/usr/bin/env python3
"""
GOT-OCR-2.0 Script
Uses stepfun-ai/GOT-OCR-2.0-hf model for advanced OCR tasks
"""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import torchvision.transforms as transforms
import argparse
import os
import time
import re
from typing import List, Dict, Any


class GOTOCR2:
    def __init__(self, model_name: str = "stepfun-ai/GOT-OCR-2.0-hf"):
        """Initialize the GOT-OCR-2.0 model"""
        print(f"Loading model: {model_name}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        # Load model and processor
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Define torchvision transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
        ])
        
        print(f"Model loaded successfully on {self.device}")
    
    def extract_text(self, image_path: str, mode: str = "plain") -> str:
        """
        Extract text from a single image
        
        Args:
            image_path: Path to the image file
            mode: OCR mode - "plain" or "formatted"
        """
        start_time = time.time()
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Process inputs based on mode
        if mode == "plain":
            # Plain text OCR
            inputs = self.processor(image_path, return_tensors="pt").to(self.device)
        elif mode == "formatted":
            # Formatted text OCR with markdown
            inputs = self.processor(image_path, return_tensors="pt", format=True).to(self.device)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'plain' or 'formatted'")
        
        # Generate text
        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs,
                do_sample=False,
                tokenizer=self.processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=4096,
                temperature=0.1
            )
        
        # Decode results
        result = self.processor.decode(
            generate_ids[0, inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        processing_time = time.time() - start_time
        print(f"Single image processing completed in {processing_time:.2f}s")
        
        return result.strip()
    
    def convert_latex_to_markdown_table(self, text: str) -> str:
        """Convert LaTeX table format to Markdown table format"""
        if '\\begin{tabular}' not in text or '\\end{tabular}' not in text:
            return text
        
        lines = text.split('\n')
        markdown_lines = []
        in_table = False
        header_added = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('\\begin{tabular}'):
                in_table = True
                continue
            elif line.startswith('\\end{tabular}'):
                in_table = False
                continue
            elif not in_table:
                continue
            elif line == '\\hline' or not line:
                continue
            
            # Process table row
            if '&' in line:
                # Remove trailing \\
                if line.endswith('\\\\'):
                    line = line[:-2].strip()
                
                # Split by & and clean up
                cells = [cell.strip() for cell in line.split('&')]
                
                # Remove \hline from the beginning of first cell if present
                if cells and cells[0].startswith('\\hline'):
                    cells[0] = cells[0].replace('\\hline', '').strip()
                
                # Create markdown row
                markdown_row = '| ' + ' | '.join(cells) + ' |'
                markdown_lines.append(markdown_row)
                
                # Add header separator after first row
                if not header_added:
                    separator = '|' + ''.join([' --- |' for _ in cells])
                    markdown_lines.append(separator)
                    header_added = True
        
        if markdown_lines:
            return '\n'.join(markdown_lines)
        else:
            return text
    
    def batch_extract_text(self, image_paths: List[str], mode: str = "plain") -> List[Dict[str, Any]]:
        """Extract text from multiple images using batched inference with batch size limit"""
        MAX_BATCH_SIZE = 5
        
        # If we have more than MAX_BATCH_SIZE images, process in chunks
        if len(image_paths) > MAX_BATCH_SIZE:
            print(f"Processing {len(image_paths)} images in batches of {MAX_BATCH_SIZE}...")
            all_results = []
            
            for i in range(0, len(image_paths), MAX_BATCH_SIZE):
                batch_paths = image_paths[i:i + MAX_BATCH_SIZE]
                batch_num = (i // MAX_BATCH_SIZE) + 1
                total_batches = (len(image_paths) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
                print(f"Processing batch {batch_num}/{total_batches} ({len(batch_paths)} images)...")
                
                batch_results = self._process_single_batch(batch_paths, mode)
                all_results.extend(batch_results)
            
            return all_results
        else:
            return self._process_single_batch(image_paths, mode)
    
    def _process_single_batch(self, image_paths: List[str], mode: str = "plain") -> List[Dict[str, Any]]:
        """Process a single batch of images (max 5 images)"""
        try:
            start_time = time.time()
            
            valid_paths = []
            
            for image_path in image_paths:
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    continue

                valid_paths.append(image_path)

            if not valid_paths:
                return []

            # Process inputs based on mode
            if mode == "plain":
                inputs = self.processor(valid_paths, return_tensors="pt").to(self.device)
            elif mode == "formatted":
                inputs = self.processor(valid_paths, return_tensors="pt", format=True).to(self.device)
            else:
                raise ValueError(f"Unknown mode: {mode}. Use 'plain' or 'formatted'")
            
            # Generate text for all images in batch
            with torch.no_grad():
                generate_ids = self.model.generate(
                    **inputs,
                    do_sample=False,
                    tokenizer=self.processor.tokenizer,
                    stop_strings="<|im_end|>",
                    max_new_tokens=4096,
                    temperature=0.1
                )
            
            # Decode results for all images
            batch_results = self.processor.batch_decode(
                generate_ids[:, inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
            processing_time = time.time() - start_time
            
            # Format results
            results = []
            for i, (image_path, extracted_text) in enumerate(zip(valid_paths, batch_results)):
                result = {
                    "image_path": image_path,
                    "extracted_text": extracted_text.strip(),
                    "processing_time": processing_time / len(valid_paths),  # Average time per image
                    "mode": mode,
                    "status": "success"
                }
                results.append(result)
            
            print(f"Batch processing completed in {processing_time:.2f}s")
            print(f"Average time per image: {processing_time/len(valid_paths):.2f}s")
            print(f"Processed {len(valid_paths)} images successfully")
            return results
            
        except Exception as e:
            print(f"Batch processing failed: {e}")
            # Fallback to individual processing
            print("Falling back to individual processing...")
            return self._individual_extract_text(image_paths, mode)
    
    def _individual_extract_text(self, image_paths: List[str], mode: str = "plain") -> List[Dict[str, Any]]:
        """Extract text from multiple images individually (fallback method)"""
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                start_time = time.time()
                
                extracted_text = self.extract_text(image_path, mode)
                
                processing_time = time.time() - start_time
                
                result = {
                    "image_path": image_path,
                    "extracted_text": extracted_text,
                    "processing_time": processing_time,
                    "mode": mode,
                    "status": "success"
                }
                
            except Exception as e:
                result = {
                    "image_path": image_path,
                    "extracted_text": "",
                    "processing_time": 0,
                    "mode": mode,
                    "status": "error",
                    "error": str(e)
                }
                print(f"Error processing {image_path}: {e}")
            
            results.append(result)
        
        return results


def natural_sort_key(text):
    """Generate a key for natural (alphanumeric) sorting"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

def main():
    parser = argparse.ArgumentParser(description="GOT-OCR-2.0 Advanced OCR")
    parser.add_argument("--image", type=str, help="Path to single image file")
    parser.add_argument("--images", type=str, nargs="+", help="Paths to multiple image files")
    parser.add_argument("--directory", type=str, help="Directory containing images")
    parser.add_argument("--mode", type=str, choices=["plain", "formatted"], 
                       default="plain", help="OCR mode")
    parser.add_argument("--output", type=str, help="Output text file for results")
    parser.add_argument("--model", type=str, default="stepfun-ai/GOT-OCR-2.0-hf", 
                       help="Model name/path")
    parser.add_argument("--raw-response", action="store_true", 
                       help="Output raw response without processing")
    
    args = parser.parse_args()
    
    # Initialize OCR model
    ocr = GOTOCR2(args.model)
    
    # Determine image paths
    image_paths = []
    
    if args.image:
        image_paths = [args.image]
    elif args.images:
        image_paths = sorted(args.images, key=natural_sort_key)
    elif args.directory:
        # Get all image files from directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        for filename in os.listdir(args.directory):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(args.directory, filename))
        # Sort paths in natural alphanumeric order
        image_paths = sorted(image_paths, key=natural_sort_key)
    else:
        parser.error("Must specify --image, --images, or --directory")
    
    if not image_paths:
        print("No images found to process")
        return
    
    print(f"Found {len(image_paths)} image(s) to process")
    print(f"OCR Mode: {args.mode}")
    
    # Print all image paths in sequence
    print("\nImage processing queue:")
    for i, image_path in enumerate(image_paths, 1):
        print(f"  {i:2d}. {image_path}")
    print()
    
    # Process images
    if len(image_paths) == 1:
        # Single image
        try:
            result = ocr.extract_text(image_paths[0], args.mode)
            
            print("\nExtracted Text:")
            print("=" * 80)
            print(result)
            print("=" * 80)
            
            if args.output:
                # Convert LaTeX table to Markdown if present unless raw response is requested
                if args.raw_response:
                    output_result = result
                else:
                    output_result = ocr.convert_latex_to_markdown_table(result)
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(output_result)
                print(f"Results saved to: {args.output}")
                
        except Exception as e:
            print(f"Error processing image: {e}")
    else:
        # Multiple images
        results = ocr.batch_extract_text(image_paths, args.mode)
        
        # Print results
        for result in results:
            print(f"\nImage: {result['image_path']}")
            print(f"Status: {result['status']}")
            print(f"Mode: {result['mode']}")
            
            if result['status'] == 'success':
                print(f"Processing time: {result['processing_time']:.2f}s")
                print("Extracted text:")
                print("-" * 60)
                print(result['extracted_text'])
                print("-" * 60)
            else:
                print(f"Error: {result['error']}")
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                if args.raw_response:
                    # Raw response mode - just write extracted text directly
                    for result in results:
                        if result['status'] == 'success':
                            f.write(result['extracted_text'])
                            f.write("\n\n")
                else:
                    # Normal processing mode
                    merged_tables = []
                    regular_text = []
                    
                    for result in results:
                        if result['status'] == 'success':
                            text = result['extracted_text']
                            # Check if this is a LaTeX table
                            if '\\begin{tabular}' in text and '\\end{tabular}' in text:
                                # Extract table content between \begin{tabular} and \end{tabular}
                                start_idx = text.find('\\begin{tabular}')
                                end_idx = text.find('\\end{tabular}') + len('\\end{tabular}')
                                table_content = text[start_idx:end_idx]
                                
                                # Extract just the table rows (between \hline entries)
                                lines = table_content.split('\n')
                                table_rows = []
                                for line in lines:
                                    if line.strip() and not line.startswith('\\begin{tabular}') and not line.startswith('\\end{tabular}') and line.strip() != '\\hline':
                                        if '&' in line:  # Only include actual data rows
                                            table_rows.append(line.strip())
                                
                                merged_tables.extend(table_rows)
                            else:
                                # Regular text, add to regular text list
                                regular_text.append(text)
                    
                    # Write regular text first
                    for text in regular_text:
                        f.write(text)
                        f.write("\n")
                    
                    # Convert and write merged table if we have table data
                    if merged_tables:
                        # Create LaTeX table string for conversion
                        latex_table = "\\begin{tabular}{|c|c|c|}\n\\hline\n"
                        for row in merged_tables:
                            latex_table += row + "\n"
                        latex_table += "\\hline\n\\end{tabular}"
                        
                        # Convert to Markdown table
                        markdown_table = ocr.convert_latex_to_markdown_table(latex_table)
                        f.write(markdown_table)
                        f.write("\n")
            
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()