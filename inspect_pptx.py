import zipfile
import re
import os

pptx_path = "Object-Detection (1).pptx"

def extract_text_from_pptx(path):
    text_content = {}
    try:
        with zipfile.ZipFile(path, 'r') as z:
            # Find all slide files
            slides = [f for f in z.namelist() if f.startswith('ppt/slides/slide') and f.endswith('.xml')]
            # Sort slides (slide1, slide2, ..., slide10 instead of 1, 10, 2)
            slides.sort(key=lambda x: int(re.search(r'slide(\d+)', x).group(1)))
            
            for slide in slides:
                try:
                    xml_content = z.read(slide).decode('utf-8')
                    # Very simple regex to find text within <a:t> tags
                    # This might miss some text or get some extra, but it's good for a quick review
                    texts = re.findall(r'<a:t>(.*?)</a:t>', xml_content)
                    slide_num = re.search(r'slide(\d+)', slide).group(1)
                    text_content[f"Slide {slide_num}"] = " ".join(texts)
                except Exception as e:
                     text_content[f"Error reading {slide}"] = str(e)
    except Exception as e:
        return f"Error opening pptx: {str(e)}"
    return text_content

if __name__ == "__main__":
    if os.path.exists(pptx_path):
        content = extract_text_from_pptx(pptx_path)
        for slide, text in content.items():
            print(f"--- {slide} ---")
            print(text.strip())
            print("\n")
    else:
        print(f"File not found: {pptx_path}")
